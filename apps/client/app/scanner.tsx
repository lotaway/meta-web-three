import React, { useState, useCallback } from 'react'
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Platform,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useRouter } from 'expo-router'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import * as ImagePicker from 'expo-image-picker'
import ScannerModule, { ScannerModuleView } from 'scanner-module'

export default function ScannerScreen() {
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const router = useRouter()

  const [scanning, setScanning] = useState(false)
  const [lastResult, setLastResult] = useState<string>('')
  const [error, setError] = useState<string>('')

  const handleScanResult = useCallback((data: string) => {
    setScanning(false)
    setLastResult(data)

    const productUrlRegex = /product\/(\d+)/
    const match = data.match(productUrlRegex)

    if (match) {
      const productId = match[1]
      Alert.alert('扫描成功', `找到商品 #${productId}`, [
        { text: '取消', style: 'cancel' },
        {
          text: '查看商品',
          onPress: () => router.push({ pathname: '/product/[id]', params: { id: productId } }),
        },
      ])
    } else {
      Alert.alert('扫描结果', data, [
        { text: '确定' },
      ])
    }
  }, [router])

  const handleScanSuccess = useCallback((event: { nativeEvent: { data: string } }) => {
    handleScanResult(event.nativeEvent.data)
  }, [handleScanResult])

  const handleError = useCallback((event: { nativeEvent: { message: string } }) => {
    setError(event.nativeEvent.message)
    setScanning(false)
    Alert.alert('扫描错误', event.nativeEvent.message)
  }, [])

  const handlePickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    })

    if (!result.canceled && result.assets[0]) {
      Alert.alert('提示', Platform.OS === 'web' ? 'Web端暂不支持图片识别，请使用摄像头扫描' : '正在识别图片...')
    }
  }

  const handleStartScan = async () => {
    setError('')
    const hasPermission = await ScannerModule.requestCameraPermissionAsync()
    if (hasPermission) {
      setScanning(true)
      setLastResult('')
    } else {
      Alert.alert('权限不足', '需要相机权限才能使用扫码功能')
    }
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>扫码</Text>
        <View style={styles.headerRight} />
      </View>

      <View style={styles.content}>
        {scanning ? (
          <View style={styles.scannerContainer}>
            <ScannerModuleView
              isScanning={scanning}
              style={styles.scannerView}
              onScanSuccess={handleScanSuccess}
              onError={handleError}
            />
            <View style={styles.scanOverlay}>
              <View style={styles.scanFrame}>
                <View style={styles.scanCorner} />
                <View style={[styles.scanCorner, styles.scanCornerTopRight]} />
                <View style={[styles.scanCorner, styles.scanCornerBottomLeft]} />
                <View style={[styles.scanCorner, styles.scanCornerBottomRight]} />
              </View>
              <Text style={styles.scanHint}>
                将二维码/条形码放入框内
              </Text>
              <TouchableOpacity
                style={styles.stopScanBtn}
                onPress={() => setScanning(false)}
              >
                <Text style={styles.stopScanText}>停止扫描</Text>
              </TouchableOpacity>
            </View>
          </View>
        ) : (
          <View style={styles.idleContainer}>
            <IconSymbol name="qrcode" size={80} color={colors.primary} />
            <Text style={[styles.idleTitle, { color: colors.fontColorDark }]}>
              扫码查找商品
            </Text>
            <Text style={[styles.idleDesc, { color: colors.fontColorLight }]}>
              扫描商品包装上的条形码或二维码
            </Text>
            <TouchableOpacity
              style={[styles.startBtn, { backgroundColor: colors.primary }]}
              onPress={handleStartScan}
            >
              <IconSymbol name="camera.viewfinder" size={20} color="#fff" />
              <Text style={styles.startBtnText}>开始扫描</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.galleryBtn, { borderColor: colors.border }]}
              onPress={handlePickImage}
            >
              <IconSymbol name="photo" size={16} color={colors.fontColorBase} />
              <Text style={[styles.galleryBtnText, { color: colors.fontColorBase }]}>
                从相册选择
              </Text>
            </TouchableOpacity>
            {lastResult && (
              <View style={[styles.resultBox, { backgroundColor: colors.card }]}>
                <Text style={[styles.resultLabel, { color: colors.fontColorLight }]}>
                  上次扫描结果
                </Text>
                <Text style={[styles.resultText, { color: colors.fontColorDark }]} numberOfLines={3}>
                  {lastResult}
                </Text>
              </View>
            )}
          </View>
        )}
      </View>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  backBtn: { padding: 8 },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginRight: 40,
  },
  headerRight: { width: 40 },
  content: { flex: 1 },
  scannerContainer: {
    flex: 1,
    backgroundColor: '#000',
    position: 'relative',
  },
  scannerView: {
    flex: 1,
  },
  scanOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanFrame: {
    width: 250,
    height: 250,
    position: 'relative',
  },
  scanCorner: {
    position: 'absolute',
    width: 40,
    height: 40,
    borderColor: '#fff',
    borderWidth: 3,
    top: 0,
    left: 0,
    borderRightWidth: 0,
    borderBottomWidth: 0,
  },
  scanCornerTopRight: {
    top: 0,
    left: 'auto',
    right: 0,
    borderRightWidth: 3,
    borderBottomWidth: 0,
    borderLeftWidth: 0,
  },
  scanCornerBottomLeft: {
    top: 'auto',
    bottom: 0,
    left: 0,
    borderRightWidth: 0,
    borderTopWidth: 0,
    borderBottomWidth: 3,
  },
  scanCornerBottomRight: {
    top: 'auto',
    bottom: 0,
    left: 'auto',
    right: 0,
    borderRightWidth: 3,
    borderTopWidth: 0,
    borderLeftWidth: 0,
    borderBottomWidth: 3,
  },
  scanHint: {
    fontSize: 14,
    marginTop: 24,
    textAlign: 'center',
    color: '#fff',
  },
  stopScanBtn: {
    position: 'absolute',
    bottom: 60,
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 25,
  },
  stopScanText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  idleContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  idleTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    marginTop: 24,
  },
  idleDesc: {
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
    lineHeight: 22,
  },
  startBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 25,
    marginTop: 40,
    gap: 8,
  },
  startBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  galleryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
    marginTop: 16,
    gap: 6,
  },
  galleryBtnText: {
    fontSize: 14,
  },
  resultBox: {
    marginTop: 40,
    padding: 16,
    borderRadius: 12,
    width: '100%',
  },
  resultLabel: {
    fontSize: 12,
    marginBottom: 8,
  },
  resultText: {
    fontSize: 14,
    lineHeight: 20,
  },
})
