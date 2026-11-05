package nl.hoverlogix.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewConfiguration
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** A box being reviewed. Coordinates are image pixels. */
class EditBox(
    val id: Int,
    var cls: Int,
    val rect: RectF,
    val score: Float,
    val fromModel: Boolean,
    val origRect: RectF = RectF(rect),
    val origCls: Int = cls,
    var rejected: Boolean = false
) {
    /** confirmed = model box kept as is, corrected = moved/resized/reclassed, rejected = wrong detection, added = missed object */
    fun status(): String = when {
        !fromModel -> "added"
        rejected -> "rejected"
        cls != origCls || moved() -> "corrected"
        else -> "confirmed"
    }

    private fun moved(): Boolean {
        val tol = 0.5f
        return abs(rect.left - origRect.left) > tol || abs(rect.top - origRect.top) > tol ||
            abs(rect.right - origRect.right) > tol || abs(rect.bottom - origRect.bottom) > tol
    }

    fun copy() = EditBox(id, cls, RectF(rect), score, fromModel, RectF(origRect), origCls, rejected)
}

/**
 * Shows a frame with editable boxes.
 *  - Tap a box to select it, drag inside to move, drag a corner handle to resize.
 *  - In "add" mode, drag on the image to draw a new box.
 *  - Two fingers: pinch to zoom, drag to pan (needed for small location labels).
 */
class BoxEditorView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    interface Listener {
        fun onSelectionChanged(box: EditBox?)
        fun onBoxesChanged()
    }

    var listener: Listener? = null
    var addMode = false
        set(v) { field = v; draft = null; invalidate() }
    var newBoxClass = 0

    private var bitmap: Bitmap? = null
    val boxes = ArrayList<EditBox>()
    var selected: EditBox? = null
        private set
    private var nextId = 1

    private val undoStack = ArrayDeque<Pair<List<EditBox>, Int?>>()

    // image -> view transform: v = i * scale + (tx, ty)
    private var scale = 1f
    private var tx = 0f
    private var ty = 0f
    private var minScale = 1f
    private val drawMatrix = Matrix()

    private val density = resources.displayMetrics.density
    private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop
    private val handleRadiusPx = 22f * density

    private val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; isAntiAlias = true }
    private val fillPaint = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
    private val textPaint = Paint().apply { color = Color.BLACK; textSize = 12f * density; isAntiAlias = true; isFakeBoldText = true }
    private val dash = DashPathEffect(floatArrayOf(10f * density, 6f * density), 0f)

    // Gesture state
    private enum class Drag { NONE, MOVE, CORNER, DRAW, PAN }
    private var drag = Drag.NONE
    private var corner = -1
    private var downX = 0f
    private var downY = 0f
    private var lastX = 0f
    private var lastY = 0f
    private var startRect = RectF()
    private var movedPastSlop = false
    private var draft: RectF? = null
    private var multiTouch = false

    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            val newScale = (scale * d.scaleFactor).coerceIn(minScale, minScale * 12f)
            val f = newScale / scale
            tx = d.focusX - (d.focusX - tx) * f
            ty = d.focusY - (d.focusY - ty) * f
            scale = newScale
            invalidate()
            return true
        }
    })

    fun setImage(bmp: Bitmap) {
        bitmap = bmp
        fitToView()
        invalidate()
    }

    fun setDetections(dets: List<Detection>) {
        boxes.clear()
        dets.forEach { boxes.add(EditBox(nextId++, it.cls, RectF(it.box), it.score, true)) }
        undoStack.clear()
        select(null)
        listener?.onBoxesChanged()
        invalidate()
    }

    fun imageWidth() = bitmap?.width ?: 0
    fun imageHeight() = bitmap?.height ?: 0

    // ---------- editing actions called by the activity ----------

    fun toggleRejectSelected() {
        val b = selected ?: return
        pushUndo()
        if (!b.fromModel) {           // a box the user added: "wrong" just removes it
            boxes.remove(b); select(null)
        } else {
            b.rejected = !b.rejected
        }
        listener?.onBoxesChanged(); invalidate()
    }

    fun setClassOfSelected(cls: Int) {
        val b = selected ?: return
        if (b.cls == cls) return
        pushUndo()
        b.cls = cls
        b.rejected = false
        listener?.onBoxesChanged(); invalidate()
    }

    fun resetSelected() {
        val b = selected ?: return
        if (!b.fromModel) return
        pushUndo()
        b.rect.set(b.origRect); b.cls = b.origCls; b.rejected = false
        listener?.onBoxesChanged(); invalidate()
    }

    fun undo(): Boolean {
        val (snap, selId) = undoStack.removeLastOrNull() ?: return false
        boxes.clear(); boxes.addAll(snap)
        select(boxes.firstOrNull { it.id == selId })
        listener?.onBoxesChanged(); invalidate()
        return true
    }

    fun canUndo() = undoStack.isNotEmpty()

    private fun pushUndo() {
        undoStack.addLast(boxes.map { it.copy() } to selected?.id)
        if (undoStack.size > 50) undoStack.removeFirst()
    }

    private fun select(b: EditBox?) {
        selected = b
        listener?.onSelectionChanged(b)
        invalidate()
    }

    // ---------- transform helpers ----------

    private fun fitToView() {
        val b = bitmap ?: return
        if (width == 0 || height == 0) return
        minScale = min(width.toFloat() / b.width, height.toFloat() / b.height)
        scale = minScale
        tx = (width - b.width * scale) / 2f
        ty = (height - b.height * scale) / 2f
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        fitToView()
    }

    private fun toImgX(x: Float) = (x - tx) / scale
    private fun toImgY(y: Float) = (y - ty) / scale
    private fun toViewRect(r: RectF) = RectF(r.left * scale + tx, r.top * scale + ty, r.right * scale + tx, r.bottom * scale + ty)

    private fun corners(r: RectF) = arrayOf(
        floatArrayOf(r.left, r.top), floatArrayOf(r.right, r.top),
        floatArrayOf(r.right, r.bottom), floatArrayOf(r.left, r.bottom)
    )

    // ---------- touch ----------

    override fun onTouchEvent(e: MotionEvent): Boolean {
        val bmp = bitmap ?: return false
        scaleDetector.onTouchEvent(e)

        when (e.actionMasked) {
            MotionEvent.ACTION_POINTER_DOWN -> {
                // Second finger: abandon any edit in progress and switch to zoom/pan.
                if (drag == Drag.MOVE || drag == Drag.CORNER) {
                    if (movedPastSlop) selected?.rect?.set(startRect)
                }
                multiTouch = true
                draft = null
                drag = Drag.PAN
                lastX = midX(e); lastY = midY(e)
                invalidate()
                return true
            }
            MotionEvent.ACTION_POINTER_UP -> return true
        }

        if (multiTouch) {
            if (e.actionMasked == MotionEvent.ACTION_MOVE && e.pointerCount >= 2) {
                val mx = midX(e); val my = midY(e)
                tx += mx - lastX; ty += my - lastY
                lastX = mx; lastY = my
                invalidate()
            }
            if (e.actionMasked == MotionEvent.ACTION_UP || e.actionMasked == MotionEvent.ACTION_CANCEL) {
                multiTouch = false; drag = Drag.NONE
            }
            return true
        }

        val ix = toImgX(e.x).coerceIn(0f, bmp.width.toFloat())
        val iy = toImgY(e.y).coerceIn(0f, bmp.height.toFloat())

        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = e.x; downY = e.y; movedPastSlop = false
                if (addMode) {
                    drag = Drag.DRAW
                    draft = RectF(ix, iy, ix, iy)
                    return true
                }
                val sel = selected
                corner = if (sel != null) hitCorner(sel, e.x, e.y) else -1
                if (sel != null && corner >= 0) {
                    drag = Drag.CORNER; startRect = RectF(sel.rect)
                } else {
                    val hit = hitTest(ix, iy)
                    if (hit != null) {
                        if (hit !== selected) select(hit)
                        drag = Drag.MOVE; startRect = RectF(hit.rect)
                    } else {
                        select(null); drag = Drag.NONE
                    }
                }
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (!movedPastSlop && hypot(e.x - downX, e.y - downY) > touchSlop) {
                    movedPastSlop = true
                    if (drag == Drag.MOVE || drag == Drag.CORNER) pushUndo()
                }
                if (!movedPastSlop) return true
                val dx = (e.x - downX) / scale
                val dy = (e.y - downY) / scale
                val sel = selected
                when (drag) {
                    Drag.DRAW -> draft?.let {
                        val sx = toImgX(downX).coerceIn(0f, bmp.width.toFloat())
                        val sy = toImgY(downY).coerceIn(0f, bmp.height.toFloat())
                        it.set(min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy))
                    }
                    Drag.MOVE -> if (sel != null) {
                        val w = startRect.width(); val h = startRect.height()
                        val l = (startRect.left + dx).coerceIn(0f, bmp.width - w)
                        val t = (startRect.top + dy).coerceIn(0f, bmp.height - h)
                        sel.rect.set(l, t, l + w, t + h)
                    }
                    Drag.CORNER -> if (sel != null) {
                        val r = RectF(startRect)
                        when (corner) {
                            0 -> { r.left = ix; r.top = iy }
                            1 -> { r.right = ix; r.top = iy }
                            2 -> { r.right = ix; r.bottom = iy }
                            3 -> { r.left = ix; r.bottom = iy }
                        }
                        sel.rect.set(min(r.left, r.right), min(r.top, r.bottom), max(r.left, r.right), max(r.top, r.bottom))
                    }
                    else -> {}
                }
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                val minSide = 4f / scale * density
                if (drag == Drag.DRAW) {
                    val d = draft
                    if (d != null && d.width() > minSide && d.height() > minSide) {
                        pushUndo()
                        val nb = EditBox(nextId++, newBoxClass, RectF(d), 1f, fromModel = false)
                        boxes.add(nb)
                        addMode = false
                        select(nb)
                        listener?.onBoxesChanged()
                    }
                    draft = null
                } else if ((drag == Drag.MOVE || drag == Drag.CORNER) && movedPastSlop) {
                    selected?.let { b ->
                        if (b.rect.width() < minSide || b.rect.height() < minSide) b.rect.set(startRect)
                    }
                    listener?.onBoxesChanged()
                }
                drag = Drag.NONE
                invalidate()
                return true
            }
        }
        return true
    }

    private fun midX(e: MotionEvent) = (e.getX(0) + e.getX(1)) / 2f
    private fun midY(e: MotionEvent) = (e.getY(0) + e.getY(1)) / 2f

    private fun hitCorner(b: EditBox, vx: Float, vy: Float): Int {
        val r = toViewRect(b.rect)
        corners(r).forEachIndexed { i, p -> if (hypot(p[0] - vx, p[1] - vy) < handleRadiusPx) return i }
        return -1
    }

    /** Smallest box under the finger wins, so a label inside a pallet is easy to pick. */
    private fun hitTest(ix: Float, iy: Float): EditBox? {
        val pad = 8f * density / scale
        return boxes.filter {
            ix >= it.rect.left - pad && ix <= it.rect.right + pad && iy >= it.rect.top - pad && iy <= it.rect.bottom + pad
        }.minByOrNull { (if (it.rejected) 1e12f else 0f) + it.rect.width() * it.rect.height() }
    }

    // ---------- drawing ----------

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bmp = bitmap ?: return
        drawMatrix.reset()
        drawMatrix.postScale(scale, scale)
        drawMatrix.postTranslate(tx, ty)
        canvas.drawBitmap(bmp, drawMatrix, bmpPaint)

        val stroke = 2.5f * density
        for (b in boxes) {
            if (b === selected) continue
            drawBox(canvas, b, stroke, false)
        }
        selected?.let { drawBox(canvas, it, stroke * 1.8f, true) }

        draft?.let {
            boxPaint.color = Labels.color(newBoxClass); boxPaint.strokeWidth = stroke; boxPaint.pathEffect = dash
            canvas.drawRect(toViewRect(it), boxPaint)
            boxPaint.pathEffect = null
        }
    }

    private fun drawBox(canvas: Canvas, b: EditBox, stroke: Float, isSelected: Boolean) {
        val r = toViewRect(b.rect)
        val color = if (b.rejected) Color.RED else Labels.color(b.cls)
        boxPaint.color = color
        boxPaint.strokeWidth = stroke
        boxPaint.alpha = if (b.rejected && !isSelected) 150 else 255
        boxPaint.pathEffect = if (b.rejected || !b.fromModel) dash else null
        canvas.drawRect(r, boxPaint)
        if (b.rejected) {
            canvas.drawLine(r.left, r.top, r.right, r.bottom, boxPaint)
            canvas.drawLine(r.right, r.top, r.left, r.bottom, boxPaint)
        }
        boxPaint.pathEffect = null
        boxPaint.alpha = 255

        val tag = when (b.status()) {
            "rejected" -> "WRONG"
            "added" -> "added"
            "corrected" -> "edited"
            else -> "${(b.score * 100).toInt()}%"
        }
        val label = "${Labels.name(b.cls)} $tag"
        val tw = textPaint.measureText(label)
        val th = textPaint.textSize
        val y = if (r.top - th - 4 > 0) r.top - th - 4 else r.top
        fillPaint.color = color
        canvas.drawRect(r.left, y, r.left + tw + 8, y + th + 4, fillPaint)
        textPaint.color = if (b.rejected) Color.WHITE else Color.BLACK
        canvas.drawText(label, r.left + 4, y + th - 1, textPaint)

        if (isSelected) {
            fillPaint.color = Color.WHITE
            boxPaint.color = color; boxPaint.strokeWidth = 2f * density
            corners(r).forEach {
                canvas.drawCircle(it[0], it[1], 9f * density, fillPaint)
                canvas.drawCircle(it[0], it[1], 9f * density, boxPaint)
            }
        }
    }
}
