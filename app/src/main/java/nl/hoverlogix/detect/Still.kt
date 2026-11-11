package nl.hoverlogix.detect

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Rect
import android.graphics.RectF

/**
 * One captured photo. High-resolution photos are kept as JPEG bytes and only the label region is
 * decoded, so a 50-megapixel burst does not exhaust memory. [width] x [height] are upright pixels.
 */
class Still private constructor(
    private val jpeg: ByteArray?, private val bitmap: Bitmap?,
    private val area: Rect, private val rotation: Int, val source: String
) {
    val width: Int = if (rotation % 180 == 0) area.width() else area.height()
    val height: Int = if (rotation % 180 == 0) area.height() else area.width()

    /** A small upright copy for the detector. Returns the bitmap and its scale relative to the full photo. */
    fun preview(maxSide: Int = 1600): Pair<Bitmap, Float> {
        if (bitmap != null) return bitmap to 1f
        var sample = 1
        while (maxOf(area.width(), area.height()) / (sample * 2) >= maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val data = jpeg!!
        val full = BitmapFactory.decodeByteArray(data, 0, data.size, opts)
        val l = (area.left / sample).coerceIn(0, full.width - 1)
        val t = (area.top / sample).coerceIn(0, full.height - 1)
        val w = (area.width() / sample).coerceIn(1, full.width - l)
        val h = (area.height() / sample).coerceIn(1, full.height - t)
        val cut = if (l == 0 && t == 0 && w == full.width && h == full.height) full else Bitmap.createBitmap(full, l, t, w, h)
        val up = LiveActivity.rotate(cut, rotation)
        return up to up.width.toFloat() / width
    }

    /** Cuts [box] (upright pixels of the full photo) plus a margin, at full resolution. */
    fun crop(box: RectF, margin: Float, minWidth: Int): Bitmap? {
        if (bitmap != null) return ScanLogic.crop(bitmap, box, margin, minWidth)
        val mx = box.width() * margin
        val my = box.height() * margin
        val l = (box.left - mx).coerceIn(0f, width - 1f)
        val t = (box.top - my).coerceIn(0f, height - 1f)
        val r = (box.right + mx).coerceIn(l + 1f, width.toFloat())
        val b = (box.bottom + my).coerceIn(t + 1f, height.toFloat())
        if (r - l < 8 || b - t < 8) return null
        // Upright rectangle -> rectangle in the stored (unrotated) JPEG.
        val aw = area.width().toFloat()
        val ah = area.height().toFloat()
        val raw = when (((rotation % 360) + 360) % 360) {
            90 -> RectF(t, ah - r, b, ah - l)
            180 -> RectF(aw - r, ah - b, aw - l, ah - t)
            270 -> RectF(aw - b, l, aw - t, r)
            else -> RectF(l, t, r, b)
        }
        val region = Rect(
            (area.left + raw.left).toInt(), (area.top + raw.top).toInt(),
            (area.left + raw.right).toInt(), (area.top + raw.bottom).toInt()
        )
        @Suppress("DEPRECATION")
        val data = jpeg!!
        val decoder = BitmapRegionDecoder.newInstance(data, 0, data.size, false) ?: return null
        val cut: Bitmap? = try {
            region.intersect(0, 0, decoder.width, decoder.height)
            decoder.decodeRegion(region, BitmapFactory.Options())
        } finally {
            decoder.recycle()
        }
        if (cut == null) return null
        val up = LiveActivity.rotate(cut, rotation)
        if (up.width >= minWidth) return up
        val s = minWidth.toFloat() / up.width
        return Bitmap.createScaledBitmap(up, minWidth, (up.height * s).toInt().coerceAtLeast(1), true)
    }

    companion object {
        fun fromJpeg(bytes: ByteArray, cropRect: Rect?, rotation: Int): Still {
            val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, o)
            val full = Rect(0, 0, o.outWidth, o.outHeight)
            val area = if (cropRect != null && !cropRect.isEmpty && full.contains(cropRect)) Rect(cropRect) else full
            return Still(bytes, null, area, rotation, "photo")
        }

        /** An already upright bitmap (a preview frame, used when no photo could be taken). */
        fun fromBitmap(bmp: Bitmap, source: String) = Still(null, bmp, Rect(0, 0, bmp.width, bmp.height), 0, source)
    }
}
