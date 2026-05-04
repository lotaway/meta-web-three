import React, { useState, useMemo } from 'react'
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  Image,
  ScrollView,
  Dimensions,
} from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'

export interface SpecOption {
  id: string
  name: string
  image?: string
  disabled?: boolean
}

export interface SpecGroup {
  id: string
  name: string
  options: SpecOption[]
}

export interface SKUInfo {
  skuId: string
  specs: { [specId: string]: string }
  price: number
  originalPrice?: number
  stock: number
  image?: string
}

interface SKUSelectorProps {
  visible: boolean
  productImage?: string
  productName?: string
  productPrice?: number
  productOriginalPrice?: number
  specs?: SpecGroup[]
  skus?: SKUInfo[]
  initialQuantity?: number
  onClose: () => void
  onConfirm: (selectedSpecs: { [specId: string]: string }, sku: SKUInfo | null, quantity: number) => void
  actionType: 'cart' | 'buy'
}

const { width: SCREEN_WIDTH } = Dimensions.get('window')

export default function SKUSelector({
  visible,
  productImage,
  productName,
  productPrice,
  productOriginalPrice,
  specs = [],
  skus = [],
  initialQuantity = 1,
  onClose,
  onConfirm,
  actionType,
}: SKUSelectorProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  const [selectedSpecs, setSelectedSpecs] = useState<{ [specId: string]: string }>({})
  const [quantity, setQuantity] = useState(initialQuantity)

  const currentSKU = useMemo(() => {
    if (skus.length === 0) return null
    return skus.find(sku => {
      return Object.entries(selectedSpecs).every(([specId, optionId]) => {
        return sku.specs[specId] === optionId
      })
    }) || null
  }, [selectedSpecs, skus])

  const displayPrice = currentSKU?.price ?? productPrice ?? 0
  const displayOriginalPrice = currentSKU?.originalPrice ?? productOriginalPrice
  const displayStock = currentSKU?.stock ?? 0

  const handleSpecSelect = (specId: string, optionId: string) => {
    setSelectedSpecs(prev => ({
      ...prev,
      [specId]: optionId,
    }))
  }

  const handleQuantityChange = (delta: number) => {
    const newQty = Math.max(1, Math.min(quantity + delta, displayStock))
    setQuantity(newQty)
  }

  const isAllSelected = specs.every(spec => selectedSpecs[spec.id])
  const isOutOfStock = displayStock <= 0

  const handleConfirm = () => {
    if (!isAllSelected) return
    onConfirm(selectedSpecs, currentSKU, quantity)
  }

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <TouchableOpacity style={styles.overlayBg} onPress={onClose} />
        <View style={[styles.bottomSheet, { backgroundColor: colors.background }]}>
          <View style={styles.header}>
            <View style={styles.productInfo}>
              {productImage && (
                <Image source={{ uri: productImage }} style={styles.productImage} />
              )}
              <View style={styles.priceSection}>
                <View style={styles.priceRow}>
                  <Text style={[styles.priceSymbol, { color: colors.primary }]}>¥</Text>
                  <Text style={[styles.priceValue, { color: colors.primary }]}>
                    {displayPrice}
                  </Text>
                </View>
                {displayOriginalPrice && displayOriginalPrice > displayPrice && (
                  <Text style={[styles.originalPrice, { color: colors.textSecondary }]}>
                    ¥{displayOriginalPrice}
                  </Text>
                )}
                <Text style={[styles.stockText, { color: colors.textSecondary }]}>
                  {t('product.sku_stock')}: {displayStock}
                </Text>
              </View>
            </View>
            <TouchableOpacity onPress={onClose}>
              <IconSymbol name="xmark" size={24} color={colors.textSecondary} />
            </TouchableOpacity>
          </View>

          <ScrollView showsVerticalScrollIndicator={false} style={styles.specsSection}>
            {specs.map(spec => (
              <View key={spec.id} style={styles.specGroup}>
                <Text style={[styles.specGroupName, { color: colors.text }]}>{spec.name}</Text>
                <View style={styles.specOptions}>
                  {spec.options.map(option => {
                    const isSelected = selectedSpecs[spec.id] === option.id
                    return (
                      <TouchableOpacity
                        key={option.id}
                        style={[
                          styles.specOption,
                          isSelected && { backgroundColor: colors.primary, borderColor: colors.primary },
                          option.disabled && styles.specOptionDisabled,
                        ]}
                        onPress={() => !option.disabled && handleSpecSelect(spec.id, option.id)}
                        disabled={option.disabled}
                      >
                        {option.image ? (
                          <Image source={{ uri: option.image }} style={styles.specOptionImage} />
                        ) : (
                          <Text
                            style={[
                              styles.specOptionText,
                              { color: isSelected ? '#fff' : colors.text },
                              option.disabled && { color: colors.textSecondary },
                            ]}
                          >
                            {option.name}
                          </Text>
                        )}
                      </TouchableOpacity>
                    )
                  })}
                </View>
              </View>
            ))}

            <View style={styles.quantitySection}>
              <Text style={[styles.quantityLabel, { color: colors.text }]}>{t('product.sku_quantity')}</Text>
              <View style={styles.quantityControl}>
                <TouchableOpacity
                  style={[styles.quantityBtn, { borderColor: colors.border }]}
                  onPress={() => handleQuantityChange(-1)}
                  disabled={quantity <= 1}
                >
                  <Text style={[styles.quantityBtnText, { color: quantity <= 1 ? colors.textSecondary : colors.text }]}>-</Text>
                </TouchableOpacity>
                <Text style={[styles.quantityValue, { color: colors.text }]}>{quantity}</Text>
                <TouchableOpacity
                  style={[styles.quantityBtn, { borderColor: colors.border }]}
                  onPress={() => handleQuantityChange(1)}
                  disabled={quantity >= displayStock}
                >
                  <Text style={[styles.quantityBtnText, { color: quantity >= displayStock ? colors.textSecondary : colors.text }]}>+</Text>
                </TouchableOpacity>
              </View>
            </View>
          </ScrollView>

          <TouchableOpacity
            style={[
              styles.confirmBtn,
              { backgroundColor: isAllSelected && !isOutOfStock ? colors.primary : colors.border },
            ]}
            onPress={handleConfirm}
            disabled={!isAllSelected || isOutOfStock}
          >
            <Text style={styles.confirmBtnText}>
              {actionType === 'cart' ? t('home.product.add_to_cart') : t('home.product.buy_now')}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  )
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  overlayBg: {
    ...StyleSheet.absoluteFillObject,
  },
  bottomSheet: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: SCREEN_WIDTH * 1.2,
    paddingBottom: 20,
  },
  header: {
    flexDirection: 'row',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  productInfo: {
    flex: 1,
    flexDirection: 'row',
  },
  productImage: {
    width: 90,
    height: 90,
    borderRadius: 8,
  },
  priceSection: {
    flex: 1,
    marginLeft: 12,
    justifyContent: 'flex-end',
  },
  priceRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  priceSymbol: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  priceValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  originalPrice: {
    fontSize: 14,
    textDecorationLine: 'line-through',
    marginTop: 4,
  },
  stockText: {
    fontSize: 12,
    marginTop: 8,
  },
  specsSection: {
    paddingHorizontal: 16,
    maxHeight: SCREEN_WIDTH * 0.8,
  },
  specGroup: {
    marginTop: 16,
  },
  specGroupName: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 12,
  },
  specOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  specOption: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    minWidth: 60,
    alignItems: 'center',
    justifyContent: 'center',
  },
  specOptionDisabled: {
    opacity: 0.4,
  },
  specOptionImage: {
    width: 36,
    height: 36,
    borderRadius: 4,
  },
  specOptionText: {
    fontSize: 14,
  },
  quantitySection: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 30,
  },
  quantityLabel: {
    fontSize: 14,
    fontWeight: '600',
    marginRight: 20,
  },
  quantityControl: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  quantityBtn: {
    width: 32,
    height: 32,
    borderWidth: 1,
    borderRadius: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  quantityBtnText: {
    fontSize: 18,
    fontWeight: '500',
  },
  quantityValue: {
    fontSize: 16,
    marginHorizontal: 16,
    minWidth: 30,
    textAlign: 'center',
  },
  confirmBtn: {
    marginHorizontal: 16,
    marginTop: 16,
    paddingVertical: 14,
    borderRadius: 24,
    alignItems: 'center',
  },
  confirmBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
})
