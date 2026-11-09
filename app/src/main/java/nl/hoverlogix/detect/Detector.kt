package nl.hoverlogix.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/** One detected object, box in pixel coordinates of the source bitmap. */
data class Detection(val cls: Int, val score: Float, val box: RectF)

/**
 * Runs the exported Ultralytics YOLO26 LiteRT/TFLite model.
 *
 * Model contract (checked against the exported files):
 *  input  [1, 3, S, S] float32 RGB 0..1 (NCHW), letterboxed with grey 114
 *  output [1, 4 + numClasses, N] : cx, cy, w, h (normalised 0..1) + per-class scores
 * NHWC input and pixel-unit boxes are also handled, so a re-export keeps working.
 */
class Detector(context: Context, assetName: String, numThreads: Int = 4) : Closeable {

    private val interpreter: Interpreter
    val inputSize: Int
    val numClasses: Int
    private val numAnchors: Int
    private val nchw: Boolean
    private val inputBuffer: ByteBuffer
    private val output: Array<Array<FloatArray>>
    private val pixels: IntArray
    private val letterbox: Bitmap
    private val canvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** Time of the last model run in milliseconds (inference only). */
    @Volatile var lastInferenceMs: Long = 0
        private set

    init {
        val fd = context.assets.openFd(assetName)
        val model = FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
        )
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(numThreads))

        val inShape = interpreter.getInputTensor(0).shape()
        nchw = inShape[1] == 3
        inputSize = if (nchw) inShape[2] else inShape[1]

        val outShape = interpreter.getOutputTensor(0).shape() // [1, 4+nc, N]
        numClasses = outShape[1] - 4
        numAnchors = outShape[2]

        inputBuffer = ByteBuffer.allocateDirect(4 * 3 * inputSize * inputSize)
            .order(ByteOrder.nativeOrder())
        output = Array(1) { Array(outShape[1]) { FloatArray(numAnchors) } }
        pixels = IntArray(inputSize * inputSize)
        letterbox = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        canvas = Canvas(letterbox)
    }

    @Synchronized
    fun detect(src: Bitmap, confThreshold: Float, iouThreshold: Float = 0.6f): List<Detection> {
        val s = inputSize.toFloat()
        val r = min(s / src.width, s / src.height)
        val nw = src.width * r
        val nh = src.height * r
        val padX = (s - nw) / 2f
        val padY = (s - nh) / 2f

        // Letterbox exactly like Ultralytics: keep aspect ratio, pad with grey 114.
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(src, null, RectF(padX, padY, padX + nw, padY + nh), paint)
        letterbox.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val area = inputSize * inputSize
        val fb = inputBuffer.asFloatBuffer()
        if (nchw) {
            for (i in 0 until area) {
                val p = pixels[i]
                fb.put(i, ((p shr 16) and 0xFF) / 255f)
                fb.put(i + area, ((p shr 8) and 0xFF) / 255f)
                fb.put(i + 2 * area, (p and 0xFF) / 255f)
            }
        } else {
            for (i in 0 until area) {
                val p = pixels[i]
                fb.put(3 * i, ((p shr 16) and 0xFF) / 255f)
                fb.put(3 * i + 1, ((p shr 8) and 0xFF) / 255f)
                fb.put(3 * i + 2, (p and 0xFF) / 255f)
            }
        }
        inputBuffer.rewind()

        val t0 = System.nanoTime()
        interpreter.run(inputBuffer, output)
        lastInferenceMs = (System.nanoTime() - t0) / 1_000_000

        val out = output[0]
        val candidates = ArrayList<Detection>()
        var maxCoord = 0f
        for (a in 0 until numAnchors) {
            var best = -1
            var bestScore = confThreshold
            for (c in 0 until numClasses) {
                val sc = out[4 + c][a]
                if (sc > bestScore) { bestScore = sc; best = c }
            }
            if (best < 0) continue
            val cx = out[0][a]; val cy = out[1][a]; val w = out[2][a]; val h = out[3][a]
            maxCoord = max(maxCoord, max(cx, cy))
            candidates.add(Detection(best, bestScore, RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        }
        // Boxes come out normalised (0..1) from the LiteRT export; scale up if so.
        val coordScale = if (maxCoord <= 2f) s else 1f

        val mapped = candidates.map { d ->
            val b = d.box
            RectF(
                ((b.left * coordScale - padX) / r).coerceIn(0f, src.width.toFloat()),
                ((b.top * coordScale - padY) / r).coerceIn(0f, src.height.toFloat()),
                ((b.right * coordScale - padX) / r).coerceIn(0f, src.width.toFloat()),
                ((b.bottom * coordScale - padY) / r).coerceIn(0f, src.height.toFloat())
            ).let { Detection(d.cls, d.score, it) }
        }
        return nms(mapped, iouThreshold)
    }

    private fun nms(dets: List<Detection>, iouThreshold: Float, maxDet: Int = 300): List<Detection> {
        val result = ArrayList<Detection>()
        for ((_, group) in dets.groupBy { it.cls }) {
            val sorted = group.sortedByDescending { it.score }.toMutableList()
            while (sorted.isNotEmpty()) {
                val best = sorted.removeAt(0)
                result.add(best)
                sorted.removeAll { iou(best.box, it.box) > iouThreshold }
            }
        }
        return result.sortedByDescending { it.score }.take(maxDet)
    }

    override fun close() = interpreter.close()

    companion object {
        fun iou(a: RectF, b: RectF): Float {
            val iw = min(a.right, b.right) - max(a.left, b.left)
            val ih = min(a.bottom, b.bottom) - max(a.top, b.top)
            if (iw <= 0f || ih <= 0f) return 0f
            val inter = iw * ih
            val union = a.width() * a.height() + b.width() * b.height() - inter
            return if (union <= 0f) 0f else inter / union
        }
    }
}

/**
 * Shared, lazily created detectors. Both are YOLOv8n v2 (7 classes, 640 px input).
 * ACCURATE = float32 (test mAP50 0.82 on DroneIMS photos), FAST = int8 (about 4x faster, mAP50 0.67:
 * weaker on small location labels).
 */
object Detectors {
    enum class Kind(val asset: String, val label: String) {
        FAST("hoverlogix_v2_640_int8.tflite", "Fast (int8)"),
        ACCURATE("hoverlogix_v2_640_float32.tflite", "Accurate")
    }

    private val cache = HashMap<Kind, Detector>()

    @Synchronized
    fun get(context: Context, kind: Kind): Detector =
        cache.getOrPut(kind) { Detector(context.applicationContext, kind.asset) }
}
