package nl.hoverlogix.detect

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
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

/** One location label offered for the scanned pallet. */
class LocationRead(val image: Bitmap, val read: LabelRead, val belongs: Boolean)

/**
 * Saves each guided scan for the DroneIMS pipeline:
 *   scans/<time>/pallet_label.jpg      zoomed, full-resolution crop of the pallet label
 *   scans/<time>/location_<n>.jpg      zoomed crops of the candidate location labels
 *   scans/<time>/scan.json             chosen location, barcodes and text read on the device
 *   scans/scans.jsonl                  one line per scan (same content as scan.json)
 * The crops can be sent on for server-side OCR and barcode decoding as they are.
 */
object ScanStore {

    fun root(context: Context): File =
        File(context.getExternalFilesDir(null) ?: context.filesDir, "scans").apply { mkdirs() }

    fun count(context: Context): Int = root(context).listFiles { f -> f.isDirectory }?.size ?: 0

    fun save(
        context: Context, locationId: String, selected: Int, locations: List<LocationRead>,
        palletLabel: Bitmap?, palletRead: LabelRead?, zoomLocation: Float, zoomPallet: Float
    ): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        val dir = File(root(context), stamp).apply { mkdirs() }
        fun write(name: String, bmp: Bitmap) =
            FileOutputStream(File(dir, name)).use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }

        val json = JSONObject().apply {
            put("time", stamp)
            put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            put("location_id", locationId)
            put("location_selected_candidate", selected)
            put("location_zoom", zoomLocation.toDouble())
            put("location_candidates", JSONArray().apply {
                locations.forEachIndexed { i, l ->
                    write("location_$i.jpg", l.image)
                    put(JSONObject().apply {
                        put("image", "location_$i.jpg")
                        put("matched_to_pallet", l.belongs)
                        put("barcodes", codesJson(l.read.codes))
                        put("text", l.read.text)
                        put("reader", l.read.engine)
                    })
                }
            })
            put("pallet_label", JSONObject().apply {
                if (palletLabel != null) {
                    write("pallet_label.jpg", palletLabel)
                    put("image", "pallet_label.jpg")
                    put("width", palletLabel.width)
                    put("height", palletLabel.height)
                }
                put("zoom", zoomPallet.toDouble())
                put("barcodes", codesJson(palletRead?.codes ?: emptyList()))
                put("text", palletRead?.text ?: "")
                put("reader", palletRead?.engine ?: "none")
            })
        }
        File(dir, "scan.json").writeText(json.toString(2))
        File(root(context), "scans.jsonl").appendText(json.toString() + "\n")
        return dir
    }

    private fun codesJson(codes: List<ReadCode>) = JSONArray().apply {
        codes.forEach { c -> put(JSONObject().apply { put("format", c.format); put("value", c.value) }) }
    }

    /** Zips all scans into the cache and returns a shareable content:// Uri. */
    fun exportZip(context: Context): Uri? {
        val root = root(context)
        if (count(context) == 0) return null
        val outDir = File(context.cacheDir, "exports").apply { mkdirs() }
        outDir.listFiles()?.forEach { it.delete() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).format(Date())
        val zip = File(outDir, "hoverlogix_scans_$stamp.zip")
        ZipOutputStream(FileOutputStream(zip)).use { zos ->
            root.walkTopDown().filter { it.isFile }.forEach { f ->
                zos.putNextEntry(ZipEntry("hoverlogix_scans/" + f.relativeTo(root).path))
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
