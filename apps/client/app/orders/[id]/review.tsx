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

type AppColors = typeof Colors.light
type SetImages = React.Dispatch<React.SetStateAction<string[]>>
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type AppRouter = typeof router

interface OrderItem {
  productId: number
  productName: string
  productPic: string
}

function buildCommentPayload(
  productId: number,
  productName: string,
  rating: number,
  content: string,
  images: string[],
  productAttr: string
) {
  return {
    xUserId: DEFAULT_USER_ID,
    commentParam: {
      productId,
      productName,
      star: rating,
      content: content.trim(),
      pics: images.join(','),
      productAttribute: productAttr || undefined,
    },
  }
}

async function submitComment(
  payload: ReturnType<typeof buildCommentPayload>,
  setSubmitting: SetBoolean,
  routerInstance: AppRouter
): Promise<void> {
  setSubmitting(true)
  try {
    await commentApi.addComment(payload)
    Alert.alert('成功', '评价提交成功', [
      { text: '确定', onPress: () => routerInstance.back() },
    ])
  } catch (error: any) {
    Alert.alert('错误', error.message || '评价提交失败')
  } finally {
    setSubmitting(false)
  }
}

async function pickReviewImage(images: string[], setImages: SetImages): Promise<void> {
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

function removeReviewImage(index: number, images: string[], setImages: SetImages) {
  setImages(images.filter((_, i) => i !== index))
}

function getRatingText(rating: number): string {
  const ratingTexts: Record<number, string> = {
    5: '非常满意',
    4: '满意',
    3: '一般',
    2: '不满意',
    1: '非常不满意',
  }
  return ratingTexts[rating] || '非常不满意'
}

function useReviewState() {
  const [rating, setRating] = useState(5)
  const [content, setContent] = useState('')
  const [images, setImages] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [productAttr, setProductAttr] = useState('')
  return { rating, setRating, content, setContent, images, setImages, submitting, setSubmitting, productAttr, setProductAttr }
}

type ReviewState = ReturnType<typeof useReviewState>

function submitReviewFromForm(productId: number, productName: string, state: ReviewState) {
  if (state.content.trim().length === 0) {
    Alert.alert('提示', '请输入评价内容')
    return
  }
  const payload = buildCommentPayload(productId, productName, state.rating, state.content, state.images, state.productAttr)
  submitComment(payload, state.setSubmitting, router)
}

function createReviewActions(productId: number, productName: string, state: ReviewState) {
  return {
    onBack: () => router.back(),
    onRate: state.setRating,
    onContentChange: state.setContent,
    onPickImage: () => pickReviewImage(state.images, state.setImages),
    onRemoveImage: (index: number) => removeReviewImage(index, state.images, state.setImages),
    onSubmit: () => submitReviewFromForm(productId, productName, state),
  }
}

type ReviewActions = ReturnType<typeof createReviewActions>

function useReviewController() {
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const params = useLocalSearchParams()
  const orderId = params.orderId ? parseInt(params.orderId as string, 10) : 0
  const productId = params.productId ? parseInt(params.productId as string, 10) : 0
  const productName = params.productName as string || ''
  const productPic = params.productPic as string || ''
  const state = useReviewState()
  return { colors, orderId, productId, productName, productPic, ...state, ...createReviewActions(productId, productName, state) }
}

interface ReviewViewProps extends ReviewActions {
  colors: AppColors
  productName: string
  productPic: string
  rating: number
  content: string
  images: string[]
  submitting: boolean
}

function ReviewHeader({ colors, onBack }: Pick<ReviewViewProps, 'colors' | 'onBack'>) {
  return (
    <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>商品评价</Text>
      <View style={styles.headerRight} />
    </View>
  )
}

function ProductInfoSection({ productPic, productName, colors }: Pick<ReviewViewProps, 'productPic' | 'productName' | 'colors'>) {
  return (
    <View style={[styles.productSection, { backgroundColor: colors.background }]}>
      <Image source={{ uri: productPic }} style={styles.productImage} />
      <View style={styles.productInfo}>
        <Text style={[styles.productName, { color: colors.fontColorDark }]} numberOfLines={2}>
          {productName}
        </Text>
      </View>
    </View>
  )
}

function StarRow({ rating, colors, onRate }: { rating: number; colors: AppColors; onRate: (star: number) => void }) {
  return (
    <>
      {[1, 2, 3, 4, 5].map((star) => (
        <TouchableOpacity key={star} onPress={() => onRate(star)} style={styles.starButton}>
          <IconSymbol
            name={star <= rating ? 'star.fill' : 'star'}
            size={36}
            color={star <= rating ? '#FFD700' : colors.fontColorDisabled}
          />
        </TouchableOpacity>
      ))}
    </>
  )
}

function RatingStars({ rating, colors, onRate }: Pick<ReviewViewProps, 'rating' | 'colors'> & { onRate: (star: number) => void }) {
  return (
    <View style={styles.starContainer}>
      <StarRow rating={rating} colors={colors} onRate={onRate} />
      <Text style={[styles.ratingText, { color: colors.fontColorDark }]}>
        {getRatingText(rating)}
      </Text>
    </View>
  )
}

function CommentTextInput({ colors, content, onContentChange }: Pick<ReviewViewProps, 'colors' | 'content' | 'onContentChange'>) {
  return (
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
      onChangeText={onContentChange}
      maxLength={500}
    />
  )
}

function CommentInputSection({ colors, content, onContentChange }: Pick<ReviewViewProps, 'colors' | 'content' | 'onContentChange'>) {
  return (
    <View style={[styles.section, { backgroundColor: colors.background }]}>
      <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>评价内容</Text>
      <CommentTextInput colors={colors} content={content} onContentChange={onContentChange} />
      <Text style={[styles.charCount, { color: colors.fontColorDisabled }]}>
        {content.length}/500
      </Text>
    </View>
  )
}

function ReviewImageItem({ uri, colors, onRemove }: { uri: string; colors: AppColors; onRemove: () => void }) {
  return (
    <View style={styles.imageItem}>
      <Image source={{ uri }} style={styles.previewImage} />
      <TouchableOpacity style={styles.removeBtn} onPress={onRemove}>
        <IconSymbol name="xmark.circle.fill" size={20} color="#fff" />
      </TouchableOpacity>
    </View>
  )
}

function AddReviewImageButton({ colors, onPress }: { colors: AppColors; onPress: () => void }) {
  return (
    <TouchableOpacity
      style={[styles.addImageBtn, { borderColor: colors.border }]}
      onPress={onPress}
    >
      <IconSymbol name="plus" size={24} color={colors.fontColorDisabled} />
      <Text style={[styles.addImageText, { color: colors.fontColorDisabled }]}>
        添加图片
      </Text>
    </TouchableOpacity>
  )
}

function ImageUploadSection({ colors, images, onPickImage, onRemoveImage }: Pick<ReviewViewProps, 'colors' | 'images' | 'onPickImage' | 'onRemoveImage'>) {
  return (
    <View style={[styles.section, { backgroundColor: colors.background }]}>
      <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>上传图片（可选）</Text>
      <View style={styles.imageGrid}>
        {images.map((uri, index) => (
          <ReviewImageItem key={index} uri={uri} colors={colors} onRemove={() => onRemoveImage(index)} />
        ))}
        {images.length < 5 && <AddReviewImageButton colors={colors} onPress={onPickImage} />}
      </View>
    </View>
  )
}

function SubmitBar({ colors, submitting, onSubmit }: Pick<ReviewViewProps, 'colors' | 'submitting' | 'onSubmit'>) {
  return (
    <View style={[styles.footer, { backgroundColor: colors.background, borderTopColor: colors.border }]}>
      <TouchableOpacity
        style={[
          styles.submitBtn,
          { backgroundColor: submitting ? colors.fontColorDisabled : colors.primary },
        ]}
        onPress={onSubmit}
        disabled={submitting}
      >
        {submitting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.submitBtnText}>提交评价</Text>
        )}
      </TouchableOpacity>
    </View>
  )
}

function ReviewView(props: ReviewViewProps) {
  const { colors } = props
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ReviewHeader colors={colors} onBack={props.onBack} />
      <ScrollView style={styles.content}>
        <ProductInfoSection productPic={props.productPic} productName={props.productName} colors={colors} />
        <View style={[styles.section, { backgroundColor: colors.background }]}>
          <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>整体评分</Text>
          <RatingStars rating={props.rating} colors={colors} onRate={props.onRate} />
        </View>
        <CommentInputSection {...props} />
        <ImageUploadSection {...props} />
      </ScrollView>
      <SubmitBar {...props} />
    </SafeAreaView>
  )
}

export default function ReviewScreen() {
  const controller = useReviewController()
  return <ReviewView {...controller} />
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
