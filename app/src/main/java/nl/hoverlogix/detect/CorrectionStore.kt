package nl.hoverlogix.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.core.content.FileProvider
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Saves reviewed frames as a ready-to-train YOLO dataset:
 *   corrections/images/<name>.jpg
 *   corrections/labels/<name>.txt   (final boxes, YOLO normalised xywh)
 *   corrections/review_log.jsonl    (what the model said vs. what the reviewer changed)
 *   corrections/classes.txt, data.yaml
 */
object CorrectionStore {

    fun root(context: Context): File =
        File(context.getExternalFilesDir(null) ?: context.filesDir, "corrections").apply { mkdirs() }

    fun count(context: Context): Int =
        File(root(context), "images").listFiles { f -> f.extension == "jpg" }?.size ?: 0

    fun save(context: Context, frame: Bitmap, boxes: List<EditBox>, source: String): String {
        val root = root(context)
        val images = File(root, "images").apply { mkdirs() }
        val labels = File(root, "labels").apply { mkdirs() }
        writeDatasetFiles(root)

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        val safeSource = source.replace(Regex("[^A-Za-z0-9_.-]"), "_").take(60)
        val name = "${stamp}_$safeSource"

        FileOutputStream(File(images, "$name.jpg")).use { frame.compress(Bitmap.CompressFormat.JPEG, 95, it) }

        val w = frame.width.toFloat()
        val h = frame.height.toFloat()
        val yolo = StringBuilder()
        boxes.filter { !it.rejected }.forEach { b ->
            val r = b.rect
            val cx = ((r.left + r.right) / 2f / w).coerceIn(0f, 1f)
            val cy = ((r.top + r.bottom) / 2f / h).coerceIn(0f, 1f)
            val bw = (r.width() / w).coerceIn(0f, 1f)
            val bh = (r.height() / h).coerceIn(0f, 1f)
            yolo.append(String.format(Locale.US, "%d %.6f %.6f %.6f %.6f\n", b.cls, cx, cy, bw, bh))
        }
        File(labels, "$name.txt").writeText(yolo.toString())

        val log = JSONObject().apply {
            put("image", "$name.jpg")
            put("source", source)
            put("time", stamp)
            put("width", frame.width)
            put("height", frame.height)
            put("boxes", JSONArray().apply {
                boxes.forEach { b ->
                    put(JSONObject().apply {
                        put("status", b.status())
                        put("class", Labels.name(b.cls))
                        put("box", rectJson(b.rect))
                        if (b.fromModel) {
                            put("model_class", Labels.name(b.origCls))
                            put("model_box", rectJson(b.origRect))
                            put("model_score", b.score.toDouble())
                        }
                    })
                }
            })
        }
        File(root, "review_log.jsonl").appendText(log.toString() + "\n")
        return name
    }

    private fun rectJson(r: android.graphics.RectF) = JSONArray().apply {
        put(r.left.toDouble()); put(r.top.toDouble()); put(r.right.toDouble()); put(r.bottom.toDouble())
    }

    private fun writeDatasetFiles(root: File) {
        File(root, "classes.txt").writeText(Labels.all().joinToString("\n") + "\n")
        val yaml = StringBuilder("path: .\ntrain: images\nval: images\nnames:\n")
        Labels.all().forEachIndexed { i, n -> yaml.append("  $i: $n\n") }
        File(root, "data.yaml").writeText(yaml.toString())
    }

    /** Zips everything into the cache and returns a shareable content:// Uri. */
    fun exportZip(context: Context): Uri? {
        val root = root(context)
        if (count(context) == 0) return null
        val outDir = File(context.cacheDir, "exports").apply { mkdirs() }
        outDir.listFiles()?.forEach { it.delete() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).format(Date())
        val zip = File(outDir, "hoverlogix_corrections_$stamp.zip")
        ZipOutputStream(FileOutputStream(zip)).use { zos ->
            root.walkTopDown().filter { it.isFile }.forEach { f ->
                zos.putNextEntry(ZipEntry("hoverlogix_corrections/" + f.relativeTo(root).path))
                f.inputStream().use { it.copyTo(zos) }
                zos.closeEntry()
            }
        }
        return FileProvider.getUriForFile(context, context.packageName + ".files", zip)
    }

    fun clear(context: Context) {
        root(context).deleteRecursively()
    }
}

/** Passes large frames between screens through a cache file instead of an Intent. */
object FrameStore {
    fun put(context: Context, bmp: Bitmap): String {
        val dir = File(context.cacheDir, "frames").apply { mkdirs() }
        dir.listFiles()?.filter { System.currentTimeMillis() - it.lastModified() > 3_600_000 }?.forEach { it.delete() }
        val f = File(dir, "frame_${System.currentTimeMillis()}.jpg")
        FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.JPEG, 97, it) }
        return f.absolutePath
    }

    fun get(path: String): Bitmap? = BitmapFactory.decodeFile(path)
}
