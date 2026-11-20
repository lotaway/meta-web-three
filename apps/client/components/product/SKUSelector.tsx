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

type Translate = (key: string) => string

function findCurrentSKU(skus: SKUInfo[], selectedSpecs: { [specId: string]: string }): SKUInfo | null {
  if (skus.length === 0) return null
  return skus.find(sku => {
    return Object.entries(selectedSpecs).every(([specId, optionId]) => {
      return sku.specs[specId] === optionId
    })
  }) || null
}

function createSpecSelectHandler(setSelectedSpecs: React.Dispatch<React.SetStateAction<{ [specId: string]: string }>>) {
  return (specId: string, optionId: string) => {
    setSelectedSpecs(prev => ({
      ...prev,
      [specId]: optionId,
    }))
  }
}

function createQuantityChangeHandler(quantity: number, displayStock: number, setQuantity: React.Dispatch<React.SetStateAction<number>>) {
  return (delta: number) => {
    const newQty = Math.max(1, Math.min(quantity + delta, displayStock))
    setQuantity(newQty)
  }
}

function createConfirmHandler(
  isAllSelected: boolean,
  selectedSpecs: { [specId: string]: string },
  currentSKU: SKUInfo | null,
  quantity: number,
  onConfirm: SKUSelectorProps['onConfirm'],
) {
  return () => {
    if (!isAllSelected) return
    onConfirm(selectedSpecs, currentSKU, quantity)
  }
}

interface SelectorState {
  selectedSpecs: { [specId: string]: string }
  quantity: number
  displayPrice: number
  displayOriginalPrice: number | undefined
  displayStock: number
  isAllSelected: boolean
  isOutOfStock: boolean
  handleSpecSelect: (specId: string, optionId: string) => void
  handleQuantityChange: (delta: number) => void
  handleConfirm: () => void
}

interface SelectorStateParams {
  initialQuantity: number
  productPrice: number | undefined
  productOriginalPrice: number | undefined
  specs: SpecGroup[]
  skus: SKUInfo[]
  onConfirm: SKUSelectorProps['onConfirm']
}

function useSKUSelectorState(params: SelectorStateParams): SelectorState {
  const { initialQuantity, productPrice, productOriginalPrice, specs, skus, onConfirm } = params
  const [selectedSpecs, setSelectedSpecs] = useState<{ [specId: string]: string }>({})
  const [quantity, setQuantity] = useState(initialQuantity)
  const currentSKU = useMemo(() => findCurrentSKU(skus, selectedSpecs), [selectedSpecs, skus])
  const displayPrice = currentSKU?.price ?? productPrice ?? 0
  const displayOriginalPrice = currentSKU?.originalPrice ?? productOriginalPrice
  const displayStock = currentSKU?.stock ?? 0
  const isAllSelected = specs.every(spec => selectedSpecs[spec.id])
  const isOutOfStock = displayStock <= 0
  const handleSpecSelect = createSpecSelectHandler(setSelectedSpecs)
  const handleQuantityChange = createQuantityChangeHandler(quantity, displayStock, setQuantity)
  const handleConfirm = createConfirmHandler(isAllSelected, selectedSpecs, currentSKU, quantity, onConfirm)
  return {
    selectedSpecs, quantity, displayPrice, displayOriginalPrice, displayStock,
    isAllSelected, isOutOfStock, handleSpecSelect, handleQuantityChange, handleConfirm,
  }
}

function SKUPrice({ displayPrice, displayOriginalPrice, displayStock, colors, t }: {
  displayPrice: number;
  displayOriginalPrice: number | undefined;
  displayStock: number;
  colors: any;
  t: Translate;
}) {
  return (
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
  )
}

function CloseButton({ colors, onClose }: { colors: any; onClose: () => void }) {
  return (
    <TouchableOpacity onPress={onClose}>
      <IconSymbol name="xmark" size={24} color={colors.textSecondary} />
    </TouchableOpacity>
  )
}

function SKUHeader({ productImage, displayPrice, displayOriginalPrice, displayStock, colors, t, onClose }: {
  productImage?: string;
  displayPrice: number;
  displayOriginalPrice: number | undefined;
  displayStock: number;
  colors: any;
  t: Translate;
  onClose: () => void;
}) {
  return (
    <View style={styles.header}>
      <View style={styles.productInfo}>
        {productImage && (
          <Image source={{ uri: productImage }} style={styles.productImage} />
        )}
        <SKUPrice
          displayPrice={displayPrice}
          displayOriginalPrice={displayOriginalPrice}
          displayStock={displayStock}
          colors={colors}
          t={t}
        />
      </View>
      <CloseButton colors={colors} onClose={onClose} />
    </View>
  )
}

function SpecOptionContent({ option, isSelected, colors }: {
  option: SpecOption;
  isSelected: boolean;
  colors: any;
}) {
  if (option.image) {
    return <Image source={{ uri: option.image }} style={styles.specOptionImage} />
  }
  return (
    <Text
      style={[
        styles.specOptionText,
        { color: isSelected ? '#fff' : colors.text },
        option.disabled && { color: colors.textSecondary },
      ]}
    >
      {option.name}
    </Text>
  )
}

function SpecOptionItem({ specId, option, isSelected, colors, onSelect }: {
  specId: string;
  option: SpecOption;
  isSelected: boolean;
  colors: any;
  onSelect: (specId: string, optionId: string) => void;
}) {
  const handlePress = () => {
    if (!option.disabled) onSelect(specId, option.id)
  }
  return (
    <TouchableOpacity
      style={[
        styles.specOption,
        isSelected && { backgroundColor: colors.primary, borderColor: colors.primary },
        option.disabled && styles.specOptionDisabled,
      ]}
      onPress={handlePress}
      disabled={option.disabled}
    >
      <SpecOptionContent option={option} isSelected={isSelected} colors={colors} />
    </TouchableOpacity>
  )
}

function SpecGroupItem({ spec, selectedSpecs, colors, onSelect }: {
  spec: SpecGroup;
  selectedSpecs: { [specId: string]: string };
  colors: any;
  onSelect: (specId: string, optionId: string) => void;
}) {
  return (
    <View style={styles.specGroup}>
      <Text style={[styles.specGroupName, { color: colors.text }]}>{spec.name}</Text>
      <View style={styles.specOptions}>
        {spec.options.map(option => (
          <SpecOptionItem
            key={option.id}
            specId={spec.id}
            option={option}
            isSelected={selectedSpecs[spec.id] === option.id}
            colors={colors}
            onSelect={onSelect}
          />
        ))}
      </View>
    </View>
  )
}

function QuantityButton({ label, delta, isDisabled, colors, onChange }: {
  label: string;
  delta: number;
  isDisabled: boolean;
  colors: any;
  onChange: (delta: number) => void;
}) {
  return (
    <TouchableOpacity
      style={[styles.quantityBtn, { borderColor: colors.border }]}
      onPress={() => onChange(delta)}
      disabled={isDisabled}
    >
      <Text style={[styles.quantityBtnText, { color: isDisabled ? colors.textSecondary : colors.text }]}>{label}</Text>
    </TouchableOpacity>
  )
}

function QuantityControl({ quantity, displayStock, colors, t, onChange }: {
  quantity: number;
  displayStock: number;
  colors: any;
  t: Translate;
  onChange: (delta: number) => void;
}) {
  return (
    <View style={styles.quantitySection}>
      <Text style={[styles.quantityLabel, { color: colors.text }]}>{t('product.sku_quantity')}</Text>
      <View style={styles.quantityControl}>
        <QuantityButton label="-" delta={-1} isDisabled={quantity <= 1} colors={colors} onChange={onChange} />
        <Text style={[styles.quantityValue, { color: colors.text }]}>{quantity}</Text>
        <QuantityButton label="+" delta={1} isDisabled={quantity >= displayStock} colors={colors} onChange={onChange} />
      </View>
    </View>
  )
}

function ConfirmButton({ isAllSelected, isOutOfStock, actionType, colors, t, onPress }: {
  isAllSelected: boolean;
  isOutOfStock: boolean;
  actionType: 'cart' | 'buy';
  colors: any;
  t: Translate;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      style={[
        styles.confirmBtn,
        { backgroundColor: isAllSelected && !isOutOfStock ? colors.primary : colors.border },
      ]}
      onPress={onPress}
      disabled={!isAllSelected || isOutOfStock}
    >
      <Text style={styles.confirmBtnText}>
        {actionType === 'cart' ? t('home.product.add_to_cart') : t('home.product.buy_now')}
      </Text>
    </TouchableOpacity>
  )
}

function SpecGroupsList({ specs, selectedSpecs, colors, onSelect }: {
  specs: SpecGroup[];
  selectedSpecs: { [specId: string]: string };
  colors: any;
  onSelect: (specId: string, optionId: string) => void;
}) {
  return (
    <>
      {specs.map(spec => (
        <SpecGroupItem
          key={spec.id}
          spec={spec}
          selectedSpecs={selectedSpecs}
          colors={colors}
          onSelect={onSelect}
        />
      ))}
    </>
  )
}

function SpecsAndQuantity({ specs, colors, t, selectedSpecs, quantity, displayStock, handleSpecSelect, handleQuantityChange }: {
  specs: SpecGroup[];
  colors: any;
  t: Translate;
} & Pick<SelectorState, 'selectedSpecs' | 'quantity' | 'displayStock' | 'handleSpecSelect' | 'handleQuantityChange'>) {
  return (
    <ScrollView showsVerticalScrollIndicator={false} style={styles.specsSection}>
      <SpecGroupsList specs={specs} selectedSpecs={selectedSpecs} colors={colors} onSelect={handleSpecSelect} />
      <QuantityControl quantity={quantity} displayStock={displayStock} colors={colors} t={t} onChange={handleQuantityChange} />
    </ScrollView>
  )
}

function SKUBottomSheetContent({
  productImage,
  specs,
  colors,
  t,
  onClose,
  actionType,
  ...state
}: {
  productImage?: string;
  specs: SpecGroup[];
  colors: any;
  t: Translate;
  onClose: () => void;
  actionType: 'cart' | 'buy';
} & SelectorState) {
  return (
    <View style={[styles.bottomSheet, { backgroundColor: colors.background }]}>
      <SKUHeader productImage={productImage} colors={colors} t={t} onClose={onClose} {...state} />
      <SpecsAndQuantity specs={specs} colors={colors} t={t} {...state} />
      <ConfirmButton actionType={actionType} colors={colors} t={t} onPress={state.handleConfirm} {...state} />
    </View>
  )
}

function SKUSelectorView({ visible, onClose, ...contentProps }: {
  visible: boolean;
  onClose: () => void;
  productImage?: string;
  specs: SpecGroup[];
  colors: any;
  t: Translate;
  actionType: 'cart' | 'buy';
} & SelectorState) {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <TouchableOpacity style={styles.overlayBg} onPress={onClose} />
        <SKUBottomSheetContent onClose={onClose} {...contentProps} />
      </View>
    </Modal>
  )
}

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
  const stateParams = { initialQuantity, productPrice, productOriginalPrice, specs, skus, onConfirm }
  const state = useSKUSelectorState(stateParams)
  return (
    <SKUSelectorView
      visible={visible}
      productImage={productImage}
      specs={specs}
      colors={colors}
      t={t}
      actionType={actionType}
      onClose={onClose}
      {...state}
    />
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
