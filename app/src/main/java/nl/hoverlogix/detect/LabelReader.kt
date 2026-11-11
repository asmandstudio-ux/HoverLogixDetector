package nl.hoverlogix.detect

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.google.zxing.BarcodeFormat
import com.google.zxing.BinaryBitmap
import com.google.zxing.DecodeHintType
import com.google.zxing.MultiFormatReader
import com.google.zxing.RGBLuminanceSource
import com.google.zxing.common.HybridBinarizer
import com.google.zxing.multi.GenericMultipleBarcodeReader
import java.util.concurrent.TimeUnit

data class ReadCode(val format: String, val value: String)

/** What was read from one label crop. [engine] says which reader found the barcodes. */
data class LabelRead(val codes: List<ReadCode>, val text: String, val engine: String) {
    /** Best single value for a location label: its barcode, else the printed text on one line. */
    fun bestValue(): String = locationId() ?: text.lines().map { it.trim() }.filter { it.isNotEmpty() }.joinToString(" ")

    /** A location ID from the barcode, or from printed text that looks like one ("AT 03 6C", "K172A"). */
    fun locationId(): String? = codes.firstOrNull()?.value ?: LOCATION_ID.find(text.uppercase())?.value?.trim()

    private companion object {
        val LOCATION_ID = Regex("(?<![A-Z0-9])[A-Z]{1,3}[ -]?\\d{1,3}[ -]?\\d{1,2}[ -]?[A-Z](?![0-9])")
    }
}

/**
 * Reads barcodes and printed text from a label crop, on the device.
 * Barcodes: ML Kit (bundled model) first, then ZXing, which is plain Java and needs no Google
 * services, so it also works on DJI controllers. Text: ML Kit Latin text recognition.
 * Call from a background thread.
 */
object LabelReader {

    private val barcodeScanner by lazy {
        BarcodeScanning.getClient(
            BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_ALL_FORMATS).build()
        )
    }
    private val textRecognizer by lazy { TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) }

    /** Loads both readers so the first scan does not pay for it. Call once from a background thread. */
    fun warmUp() {
        val blank = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        runCatching { mlKitCodes(blank) }
        runCatching { mlKitText(blank) }
    }

    fun read(bmp: Bitmap, wantText: Boolean = true): LabelRead {
        var engine = "mlkit"
        val ml = runCatching { mlKitCodes(bmp) }
        var codes = ml.getOrDefault(emptyList())
        // ZXing only when ML Kit cannot run on this device (no Google services); running both doubles the time.
        if (ml.isFailure) {
            codes = runCatching { zxingCodes(bmp) }.getOrDefault(emptyList())
            engine = "zxing"
        }
        if (codes.isEmpty()) engine = "none"
        val text = if (wantText) runCatching { mlKitText(bmp) }.getOrDefault("") else ""
        return LabelRead(codes.distinct(), text, engine)
    }

    private fun mlKitCodes(bmp: Bitmap): List<ReadCode> {
        val found = Tasks.await(barcodeScanner.process(InputImage.fromBitmap(bmp, 0)), 5, TimeUnit.SECONDS)
        return found.mapNotNull { b -> b.rawValue?.let { ReadCode(formatName(b.format), it) } }
    }

    private fun mlKitText(bmp: Bitmap): String =
        Tasks.await(textRecognizer.process(InputImage.fromBitmap(bmp, 0)), 6, TimeUnit.SECONDS).text

    private fun zxingCodes(src: Bitmap): List<ReadCode> {
        // ZXing is slow on very large images; 1800 px on the long side keeps label barcodes readable.
        val longSide = maxOf(src.width, src.height)
        val bmp = if (longSide > 1800) {
            val s = 1800f / longSide
            Bitmap.createScaledBitmap(src, (src.width * s).toInt(), (src.height * s).toInt(), true)
        } else src
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        val image = BinaryBitmap(HybridBinarizer(RGBLuminanceSource(bmp.width, bmp.height, px)))
        val hints: Map<DecodeHintType, Any> = mapOf(
            DecodeHintType.TRY_HARDER to true,
            DecodeHintType.POSSIBLE_FORMATS to listOf(
                BarcodeFormat.CODE_128, BarcodeFormat.CODE_39, BarcodeFormat.ITF, BarcodeFormat.EAN_13,
                BarcodeFormat.QR_CODE, BarcodeFormat.DATA_MATRIX
            )
        )
        return try {
            GenericMultipleBarcodeReader(MultiFormatReader()).decodeMultiple(image, hints)
                .map { ReadCode(it.barcodeFormat.name, it.text) }
        } catch (e: com.google.zxing.NotFoundException) {
            emptyList()
        }
    }

    private fun formatName(f: Int): String = when (f) {
        Barcode.FORMAT_CODE_128 -> "CODE_128"
        Barcode.FORMAT_CODE_39 -> "CODE_39"
        Barcode.FORMAT_CODE_93 -> "CODE_93"
        Barcode.FORMAT_CODABAR -> "CODABAR"
        Barcode.FORMAT_DATA_MATRIX -> "DATA_MATRIX"
        Barcode.FORMAT_EAN_13 -> "EAN_13"
        Barcode.FORMAT_EAN_8 -> "EAN_8"
        Barcode.FORMAT_ITF -> "ITF"
        Barcode.FORMAT_QR_CODE -> "QR_CODE"
        Barcode.FORMAT_UPC_A -> "UPC_A"
        Barcode.FORMAT_UPC_E -> "UPC_E"
        Barcode.FORMAT_PDF417 -> "PDF417"
        Barcode.FORMAT_AZTEC -> "AZTEC"
        else -> "UNKNOWN"
    }
}
