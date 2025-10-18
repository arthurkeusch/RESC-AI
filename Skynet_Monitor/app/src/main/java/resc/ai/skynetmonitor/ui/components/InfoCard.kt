package resc.ai.skynetmonitor.ui.components

import android.annotation.SuppressLint
import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun InfoCard(
    label: String,
    value: String,
    history: List<Float> = emptyList(),
    color: Color = Color(0xFF64B5F6),
    minValue: Float = 0f,
    maxValue: Float = 100f
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(label, fontWeight = FontWeight.Medium, fontSize = 16.sp)
                Text(value, fontWeight = FontWeight.Bold, fontSize = 16.sp)
            }
            if (history.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                MiniGraph(
                    data = history,
                    color = color,
                    minValue = minValue,
                    maxValue = maxValue
                )
            }
        }
    }
}

@SuppressLint("DefaultLocale")
@Composable
fun MiniGraph(
    data: List<Float>,
    color: Color,
    minValue: Float,
    maxValue: Float,
    modifier: Modifier = Modifier
        .fillMaxWidth()
        .height(80.dp)
) {
    val density = LocalDensity.current
    val leftPaddingPx = with(density) { 36.dp.toPx() }
    val labelTextSizePx = with(density) { 12.sp.toPx() }
    val labelColorArgb = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f).toArgb()
    val paint = Paint().apply {
        this.color = labelColorArgb
        textSize = labelTextSizePx
        textAlign = Paint.Align.RIGHT
        isAntiAlias = true
    }
    val ySteps = 5

    Canvas(modifier = modifier) {
        if (data.isEmpty()) return@Canvas

        val contentLeft = leftPaddingPx
        val contentWidth = size.width - contentLeft
        val contentHeight = size.height

        val padded = if (data.size < 60) List(60 - data.size) { data.first() } + data else data
        val normalized = padded.map {
            ((it - minValue) / (maxValue - minValue).coerceAtLeast(0.0001f)).coerceIn(0f, 1f)
        }
        val stepX = contentWidth / (normalized.size - 1).coerceAtLeast(1)
        val stepY = contentHeight / ySteps

        for (i in 0..ySteps) {
            val y = i * stepY
            drawLine(
                color = Color.Gray.copy(alpha = 0.15f),
                start = Offset(contentLeft, y),
                end = Offset(size.width, y)
            )
            val tick = minValue + (ySteps - i) * (maxValue - minValue) / ySteps
            drawContext.canvas.nativeCanvas.drawText(
                String.format("%.0f", tick),
                contentLeft - 8f,
                y + paint.textSize / 3,
                paint
            )
        }

        val verticalLines = 6
        val stepXGrid = contentWidth / verticalLines
        repeat(verticalLines + 1) {
            val x = contentLeft + it * stepXGrid
            drawLine(
                color = Color.Gray.copy(alpha = 0.08f),
                start = Offset(x, 0f),
                end = Offset(x, contentHeight)
            )
        }

        val path = Path()
        normalized.forEachIndexed { i, v ->
            val x = contentLeft + i * stepX
            val y = contentHeight - (v * contentHeight)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path = path, color = color, style = Stroke(width = 3f))
    }
}
