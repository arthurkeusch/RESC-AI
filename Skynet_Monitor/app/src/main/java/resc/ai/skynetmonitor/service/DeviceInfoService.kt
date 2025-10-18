package resc.ai.skynetmonitor.service

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager

object DeviceInfoService {

    @SuppressLint("DefaultLocale")
    fun getStaticHardwareInfo(context: Context): Map<String, String> {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)

        return mapOf(
            "Brand" to Build.MANUFACTURER,
            "Model" to Build.MODEL,
            "Android Version" to Build.VERSION.RELEASE,
            "CPU Architecture" to Build.SUPPORTED_ABIS.firstOrNull().orEmpty(),
            "CPU Cores" to Runtime.getRuntime().availableProcessors().toString(),
            "SoC" to Build.HARDWARE,
            "GPU" to getGpuName(),
            "RAM" to String.format("%.2f GB", memInfo.totalMem / (1024.0 * 1024 * 1024))
        )
    }

    @SuppressLint("DefaultLocale")
    fun getDynamicSystemState(context: Context): Map<String, String> {
        val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val batteryTemp =
            intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)?.toDouble()?.div(10.0) ?: 0.0

        val ramInfo = getRamUsage(context)

        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val thermalStatus = when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
            PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
            PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
            PowerManager.THERMAL_STATUS_LIGHT -> "Light"
            else -> "Normal"
        }

        return mapOf(
            "Thermal Status" to thermalStatus,
            "Battery Temperature" to String.format("%.1f°C", batteryTemp),
            "RAM Usage" to ramInfo,
            "Battery Level" to "$level%"
        )
    }

    private fun getGpuName(): String {
        return try {
            val display: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val version = IntArray(2)
            EGL14.eglInitialize(display, version, 0, version, 1)
            val configAttribs = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            EGL14.eglChooseConfig(display, configAttribs, 0, configs, 0, 1, numConfigs, 0)
            val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
            val contextGL: EGLContext =
                EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
            val surfaceAttribs = intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE)
            val surface: EGLSurface =
                EGL14.eglCreatePbufferSurface(display, configs[0], surfaceAttribs, 0)
            EGL14.eglMakeCurrent(display, surface, surface, contextGL)

            val renderer = GLES20.glGetString(GLES20.GL_RENDERER)
            val vendor = GLES20.glGetString(GLES20.GL_VENDOR)

            EGL14.eglMakeCurrent(
                display,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_CONTEXT
            )
            EGL14.eglDestroySurface(display, surface)
            EGL14.eglDestroyContext(display, contextGL)
            EGL14.eglTerminate(display)

            when {
                renderer != null && vendor != null -> "$renderer ($vendor)"
                renderer != null -> renderer
                vendor != null -> vendor
                else -> "Unknown GPU"
            }
        } catch (_: Exception) {
            "Unknown GPU"
        }
    }

    @SuppressLint("DefaultLocale")
    private fun getRamUsage(context: Context): String {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        val used = (memInfo.totalMem - memInfo.availMem) / (1024 * 1024 * 1024.0)
        val total = memInfo.totalMem / (1024 * 1024 * 1024.0)
        return String.format("%.2f / %.0f GB", used, total)
    }
}
