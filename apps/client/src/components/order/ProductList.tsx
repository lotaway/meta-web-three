import React from 'react'
import { View, Text, Image, TouchableOpacity } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { useRouter } from 'expo-router'

type AppColors = typeof Colors.light

interface ProductListProps {
  orderItems: any[]
  status: number
  onReview: (item: any) => void
}

function ProductRowContent({ item, colors, onOpen }: { item: any; colors: AppColors; onOpen: () => void }) {
  return (
    <TouchableOpacity style={styles.productRowContent} onPress={onOpen}>
      <Image source={{ uri: item.productPic || '' }} style={styles.productImage} />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
          {item.productName}
        </Text>
        <View style={styles.productFooter}>
          <Text style={[styles.productPrice, { color: colors.primary }]}>¥{item.productPrice}</Text>
          <Text style={[styles.productQty, { color: colors.textSecondary }]}>x{item.productQuantity}</Text>
        </View>
      </View>
    </TouchableOpacity>
  )
}

function ReviewItemButton({ colors, onPress }: { colors: AppColors; onPress: () => void }) {
  return (
    <TouchableOpacity style={styles.reviewItemBtn} onPress={onPress}>
      <IconSymbol name="pencil" size={16} color={colors.primary} />
      <Text style={[styles.reviewItemText, { color: colors.primary }]}>评价</Text>
    </TouchableOpacity>
  )
}

function ProductRow({ item, status, colors, onReview, onOpen }: {
  item: any
  status: number
  colors: AppColors
  onReview: (item: any) => void
  onOpen: () => void
}) {
  return (
    <View style={[styles.productRow, { borderBottomColor: colors.border }]}>
      <ProductRowContent item={item} colors={colors} onOpen={onOpen} />
      {status === 3 && <ReviewItemButton colors={colors} onPress={() => onReview(item)} />}
    </View>
  )
}

export function ProductList({ orderItems, status, onReview }: ProductListProps) {
  const { t } = useTranslation()
  const colors = Colors[useColorScheme() ?? 'light']
  const router = useRouter()
  const openProduct = (item: any) => router.push({ pathname: '/product/[id]', params: { id: item.productId } })
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.products_title')}</Text>
      {orderItems?.map((item: any, index: number) => (
        <ProductRow key={index} item={item} status={status} colors={colors} onReview={onReview} onOpen={() => openProduct(item)} />
      ))}
    </View>
  )
}

const styles = {
  section: {
    marginBottom: 10,
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    marginBottom: 12,
  },
  productRow: {
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  productRowContent: { flexDirection: 'row' as const, flex: 1 },
  reviewItemBtn: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#eee',
    backgroundColor: '#fff',
    alignSelf: 'center' as const,
    gap: 4,
  },
  reviewItemText: {
    fontSize: 12,
    fontWeight: '500' as const,
  },
  productImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
  },
  productInfo: {
    flex: 1,
    marginLeft: 12,
    justifyContent: 'space-between' as const,
  },
  productName: { fontSize: 14, lineHeight: 20 },
  productFooter: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    alignItems: 'center' as const,
    marginTop: 8,
  },
  productPrice: { fontSize: 16, fontWeight: 'bold' as const },
  productQty: { fontSize: 14 },
}
