import React, { useState } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  Image,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { router, useLocalSearchParams } from 'expo-router'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { commentApi, DEFAULT_USER_ID } from '@/api/generated'
import * as ImagePicker from 'expo-image-picker'

interface OrderItem {
  productId: number
  productName: string
  productPic: string
}

export default function ReviewScreen() {
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const params = useLocalSearchParams()

  const orderId = params.orderId ? parseInt(params.orderId as string, 10) : 0
  const productId = params.productId ? parseInt(params.productId as string, 10) : 0
  const productName = params.productName as string || ''
  const productPic = params.productPic as string || ''

  const [rating, setRating] = useState(5)
  const [content, setContent] = useState('')
  const [images, setImages] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [productAttr, setProductAttr] = useState('')

  const handlePickImage = async () => {
    if (images.length >= 5) {
      Alert.alert('提示', '最多上传5张图片')
      return
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    })

    if (!result.canceled && result.assets[0]) {
      setImages([...images, result.assets[0].uri])
    }
  }

  const handleRemoveImage = (index: number) => {
    setImages(images.filter((_, i) => i !== index))
  }

  const handleSubmit = async () => {
    if (content.trim().length === 0) {
      Alert.alert('提示', '请输入评价内容')
      return
    }

    setSubmitting(true)
    try {
      await commentApi.addComment({
        xUserId: DEFAULT_USER_ID,
        commentParam: {
          productId,
          productName,
          star: rating,
          content: content.trim(),
          pics: images.join(','),
          productAttribute: productAttr || undefined,
        },
      })

      Alert.alert('成功', '评价提交成功', [
        { text: '确定', onPress: () => router.back() },
      ])
    } catch (error: any) {
      Alert.alert('错误', error.message || '评价提交失败')
    } finally {
      setSubmitting(false)
    }
  }

  const renderStars = () => (
    <View style={styles.starContainer}>
      {[1, 2, 3, 4, 5].map((star) => (
        <TouchableOpacity
          key={star}
          onPress={() => setRating(star)}
          style={styles.starButton}
        >
          <IconSymbol
            name={star <= rating ? 'star.fill' : 'star'}
            size={36}
            color={star <= rating ? '#FFD700' : colors.fontColorDisabled}
          />
        </TouchableOpacity>
      ))}
      <Text style={[styles.ratingText, { color: colors.fontColorDark }]}>
        {rating === 5 ? '非常满意' : rating === 4 ? '满意' : rating === 3 ? '一般' : rating === 2 ? '不满意' : '非常不满意'}
      </Text>
    </View>
  )

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>商品评价</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView style={styles.content}>
        {/* 商品信息 */}
        <View style={[styles.productSection, { backgroundColor: colors.background }]}>
          <Image source={{ uri: productPic }} style={styles.productImage} />
          <View style={styles.productInfo}>
            <Text style={[styles.productName, { color: colors.fontColorDark }]} numberOfLines={2}>
              {productName}
            </Text>
          </View>
        </View>

        {/* 评分 */}
        <View style={[styles.section, { backgroundColor: colors.background }]}>
          <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>
            整体评分
          </Text>
          {renderStars()}
        </View>

        {/* 评价内容 */}
        <View style={[styles.section, { backgroundColor: colors.background }]}>
          <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>
            评价内容
          </Text>
          <TextInput
            style={[styles.textInput, {
              color: colors.fontColorDark,
              borderColor: colors.border,
              backgroundColor: colors.background,
            }]}
            multiline
            numberOfLines={6}
            placeholder="分享你对商品的真实感受..."
            placeholderTextColor={colors.fontColorDisabled}
            value={content}
            onChangeText={setContent}
            maxLength={500}
          />
          <Text style={[styles.charCount, { color: colors.fontColorDisabled }]}>
            {content.length}/500
          </Text>
        </View>

        {/* 上传图片 */}
        <View style={[styles.section, { backgroundColor: colors.background }]}>
          <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>
            上传图片（可选）
          </Text>
          <View style={styles.imageGrid}>
            {images.map((uri, index) => (
              <View key={index} style={styles.imageItem}>
                <Image source={{ uri }} style={styles.previewImage} />
                <TouchableOpacity
                  style={styles.removeBtn}
                  onPress={() => handleRemoveImage(index)}
                >
                  <IconSymbol name="xmark.circle.fill" size={20} color="#fff" />
                </TouchableOpacity>
              </View>
            ))}
            {images.length < 5 && (
              <TouchableOpacity
                style={[styles.addImageBtn, { borderColor: colors.border }]}
                onPress={handlePickImage}
              >
                <IconSymbol name="plus" size={24} color={colors.fontColorDisabled} />
                <Text style={[styles.addImageText, { color: colors.fontColorDisabled }]}>
                  添加图片
                </Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      </ScrollView>

      {/* 提交按钮 */}
      <View style={[styles.footer, { backgroundColor: colors.background, borderTopColor: colors.border }]}>
        <TouchableOpacity
          style={[
            styles.submitBtn,
            { backgroundColor: submitting ? colors.fontColorDisabled : colors.primary },
          ]}
          onPress={handleSubmit}
          disabled={submitting}
        >
          {submitting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.submitBtnText}>提交评价</Text>
          )}
        </TouchableOpacity>
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
  productSection: {
    flexDirection: 'row',
    padding: 16,
    marginBottom: 10,
  },
  productImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  productInfo: {
    flex: 1,
    marginLeft: 12,
    justifyContent: 'center',
  },
  productName: {
    fontSize: 14,
    lineHeight: 20,
  },
  section: {
    padding: 16,
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  starContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  starButton: {
    padding: 4,
  },
  ratingText: {
    fontSize: 14,
    marginLeft: 12,
  },
  textInput: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    minHeight: 120,
    textAlignVertical: 'top',
  },
  charCount: {
    fontSize: 12,
    textAlign: 'right',
    marginTop: 8,
  },
  imageGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  imageItem: {
    width: 100,
    height: 100,
    borderRadius: 8,
    overflow: 'hidden',
    position: 'relative',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  removeBtn: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 10,
  },
  addImageBtn: {
    width: 100,
    height: 100,
    borderRadius: 8,
    borderWidth: 1,
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
  },
  addImageText: {
    fontSize: 12,
    marginTop: 4,
  },
  footer: {
    padding: 16,
    borderTopWidth: 1,
  },
  submitBtn: {
    paddingVertical: 14,
    borderRadius: 25,
    alignItems: 'center',
  },
  submitBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
})
