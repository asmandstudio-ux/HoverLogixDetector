package nl.hoverlogix.detect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * Draws detections on top of a camera preview or video.
 * FILL = source is centre-cropped to the view (PreviewView FILL_CENTER),
 * FIT  = source is letterboxed inside the view.
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    enum class ScaleMode { FILL, FIT }

    var scaleMode = ScaleMode.FILL
    private var detections: List<Detection> = emptyList()
    private var srcW = 1
    private var srcH = 1

    private val density = resources.displayMetrics.density
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 2.5f * density; isAntiAlias = true }
    private val textBg = Paint().apply { style = Paint.Style.FILL }
    private val textPaint = Paint().apply {
        color = Color.BLACK; textSize = 11f * density; isAntiAlias = true; isFakeBoldText = true
    }

    fun setDetections(dets: List<Detection>, sourceWidth: Int, sourceHeight: Int) {
        detections = dets
        srcW = max(1, sourceWidth)
        srcH = max(1, sourceHeight)
        invalidate()
    }

    fun clear() { detections = emptyList(); invalidate() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return
        val sx = width.toFloat() / srcW
        val sy = height.toFloat() / srcH
        val s = if (scaleMode == ScaleMode.FILL) max(sx, sy) else min(sx, sy)
        val ox = (width - srcW * s) / 2f
        val oy = (height - srcH * s) / 2f

        for (d in detections) {
            val c = Labels.color(d.cls)
            boxPaint.color = c
            val r = RectF(d.box.left * s + ox, d.box.top * s + oy, d.box.right * s + ox, d.box.bottom * s + oy)
            canvas.drawRect(r, boxPaint)
            val label = "${Labels.name(d.cls)} ${(d.score * 100).toInt()}%"
            val tw = textPaint.measureText(label)
            val th = textPaint.textSize
            val ty = if (r.top - th - 4 > 0) r.top - th - 4 else r.top
            textBg.color = c
            canvas.drawRect(r.left, ty, r.left + tw + 8, ty + th + 4, textBg)
            canvas.drawText(label, r.left + 4, ty + th - 1, textPaint)
        }
    }
}
