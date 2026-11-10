plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "nl.hoverlogix.detect"
    compileSdk = 34

    defaultConfig {
        applicationId = "nl.hoverlogix.detect"
        minSdk = 26          // Android 8.0+; DJI RC Plus 2 and current phones are fine
        targetSdk = 34
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        versionName = "1.0." + (System.getenv("GITHUB_RUN_NUMBER") ?: "0")
    }

    // Fixed signing key so every CI build installs over the previous one.
    signingConfigs {
        getByName("debug") {
            storeFile = file("hoverlogix-dev.keystore")
            storePassword = "hoverlogix"
            keyAlias = "hoverlogix"
            keyPassword = "hoverlogix"
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("debug")
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // The .tflite model must stay uncompressed so it can be memory-mapped.
    androidResources { noCompress += "tflite" }
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // Bundled LiteRT (TensorFlow Lite) runtime, Interpreter API in package org.tensorflow.lite.
    // No Google Play Services needed, so it also runs on DJI controllers.
    // If Gradle cannot resolve this version, Android Studio will suggest the latest one.
    implementation("com.google.ai.edge.litert:litert:2.1.0")

    // Guided scan: read barcodes and text from the zoomed label crops on the device.
    // ML Kit with bundled models (no model download), ZXing as a pure-Java barcode fallback
    // for devices without Google services such as DJI controllers.
    implementation("com.google.mlkit:barcode-scanning:17.3.0")
    implementation("com.google.mlkit:text-recognition:16.0.1")
    implementation("com.google.zxing:core:3.5.3")
}
