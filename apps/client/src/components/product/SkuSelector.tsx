import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Modal, StyleSheet, ScrollView } from 'react-native';
import { useTranslation } from 'react-i18next';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

export interface SpecOption {
  id: string;
  name: string;
}

export interface SpecGroup {
  id: string;
  name: string;
  options: SpecOption[];
}

export interface SKUInfo {
  skuId: string;
  specs: string;
  price: number;
  stock: number;
}

interface SkuSelectorProps {
  visible: boolean;
  productImage?: string;
  productName?: string;
  productPrice?: string;
  productOriginalPrice?: string;
  specs: SpecGroup[];
  skus: SKUInfo[];
  onClose: () => void;
  onConfirm: (selectedSpecs: Record<string, string>, sku: SKUInfo | null, quantity: number) => void;
  actionType?: 'cart' | 'buy';
}

function SpecOptionButton({ option, selected, colors, onSelect }: {
  option: SpecOption;
  selected: boolean;
  colors: any;
  onSelect: () => void;
}) {
  return (
    <TouchableOpacity
      style={[styles.option, selected && { backgroundColor: colors.primary }]}
      onPress={onSelect}
    >
      <Text style={[
        styles.optionText,
        { color: selected ? '#fff' : colors.fontColorDark }
      ]}>
        {option.name}
      </Text>
    </TouchableOpacity>
  );
}

function SpecGroupView({ group, selectedSpecs, colors, onSelectSpec }: {
  group: SpecGroup;
  selectedSpecs: Record<string, string>;
  colors: any;
  onSelectSpec: (groupId: string, optionId: string) => void;
}) {
  return (
    <View style={styles.specGroup}>
      <Text style={[styles.specName, { color: colors.fontColorDark }]}>{group.name}</Text>
      <View style={styles.options}>
        {group.options.map(option => (
          <SpecOptionButton
            key={option.id}
            option={option}
            selected={selectedSpecs[group.id] === option.id}
            colors={colors}
            onSelect={() => onSelectSpec(group.id, option.id)}
          />
        ))}
      </View>
    </View>
  );
}

function ProductHeader({ productImage, productPrice, productName, colors, onClose }: {
  productImage?: string;
  productPrice?: string;
  productName?: string;
  colors: any;
  onClose: () => void;
}) {
  return (
    <View style={styles.header}>
      {productImage && (
        <View style={styles.imageContainer}>
          <View style={styles.imagePlaceholder} />
        </View>
      )}
      <View style={styles.info}>
        <Text style={[styles.price, { color: colors.primary }]}>¥{productPrice}</Text>
        <Text style={[styles.name, { color: colors.fontColorDark }]} numberOfLines={2}>{productName}</Text>
      </View>
      <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
        <IconSymbol name="xmark" size={20} color={colors.fontColorBase} />
      </TouchableOpacity>
    </View>
  );
}

function ConfirmFooter({ actionType, colors, onConfirm }: {
  actionType: 'cart' | 'buy';
  colors: any;
  onConfirm: () => void;
}) {
  const { t } = useTranslation();
  return (
    <View style={styles.footer}>
      <TouchableOpacity
        style={[styles.confirmBtn, { backgroundColor: actionType === 'buy' ? colors.primary : '#ff9500' }]}
        onPress={onConfirm}
      >
        <Text style={styles.confirmText}>
          {t(actionType === 'buy' ? 'home.product.buy_now' : 'home.product.add_to_cart')}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

function findMatchingSKU(skus: SKUInfo[]): SKUInfo | null {
  if (skus.length === 0) return null;
  return skus[0];
}

function createSpecSelectHandler(setSelectedSpecs: React.Dispatch<React.SetStateAction<Record<string, string>>>) {
  return (groupId: string, optionId: string) => {
    setSelectedSpecs(prev => ({ ...prev, [groupId]: optionId }));
  };
}

function createConfirmHandler(
  selectedSpecs: Record<string, string>,
  skus: SKUInfo[],
  quantity: number,
  onConfirm: SkuSelectorProps['onConfirm'],
) {
  return () => {
    const sku = findMatchingSKU(skus);
    onConfirm(selectedSpecs, sku, quantity);
  };
}

function useSkuSelectorState(skus: SKUInfo[], onConfirm: SkuSelectorProps['onConfirm']) {
  const [selectedSpecs, setSelectedSpecs] = useState<Record<string, string>>({});
  const [quantity, setQuantity] = useState(1);
  const handleSelectSpec = createSpecSelectHandler(setSelectedSpecs);
  const handleConfirm = createConfirmHandler(selectedSpecs, skus, quantity, onConfirm);
  return { selectedSpecs, handleSelectSpec, handleConfirm };
}

function SpecsGroupList({ specs, selectedSpecs, colors, onSelectSpec }: {
  specs: SpecGroup[];
  selectedSpecs: Record<string, string>;
  colors: any;
  onSelectSpec: (groupId: string, optionId: string) => void;
}) {
  return (
    <ScrollView style={styles.specsContainer}>
      {specs.map(group => (
        <SpecGroupView key={group.id} group={group} selectedSpecs={selectedSpecs} colors={colors} onSelectSpec={onSelectSpec} />
      ))}
    </ScrollView>
  );
}

function SkuSelectorContent({
  productImage,
  productName,
  productPrice,
  specs,
  selectedSpecs,
  colors,
  onClose,
  onSelectSpec,
  actionType,
  onConfirm,
}: {
  productImage?: string;
  productName?: string;
  productPrice?: string;
  specs: SpecGroup[];
  selectedSpecs: Record<string, string>;
  colors: any;
  onClose: () => void;
  onSelectSpec: (groupId: string, optionId: string) => void;
  actionType: 'cart' | 'buy';
  onConfirm: () => void;
}) {
  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <ProductHeader
        productImage={productImage}
        productPrice={productPrice}
        productName={productName}
        colors={colors}
        onClose={onClose}
      />
      <SpecsGroupList specs={specs} selectedSpecs={selectedSpecs} colors={colors} onSelectSpec={onSelectSpec} />
      <ConfirmFooter actionType={actionType} colors={colors} onConfirm={onConfirm} />
    </View>
  );
}

type SkuSelectorModalProps = Pick<SkuSelectorProps, 'visible' | 'productImage' | 'productName' | 'productPrice' | 'specs'> & {
  actionType: 'cart' | 'buy';
  colors: any;
  onClose: () => void;
  selectedSpecs: Record<string, string>;
  onSelectSpec: (groupId: string, optionId: string) => void;
  onConfirm: () => void;
}

function SkuSelectorModal({
  visible, productImage, productName, productPrice, specs, actionType,
  colors, onClose, selectedSpecs, onSelectSpec, onConfirm,
}: SkuSelectorModalProps) {
  return (
    <Modal visible={visible} animationType="slide" transparent>
      <View style={styles.overlay}>
        <SkuSelectorContent
          productImage={productImage}
          productName={productName}
          productPrice={productPrice}
          specs={specs}
          selectedSpecs={selectedSpecs}
          colors={colors}
          onClose={onClose}
          onSelectSpec={onSelectSpec}
          actionType={actionType}
          onConfirm={onConfirm}
        />
      </View>
    </Modal>
  );
}

export function SkuSelector({
  visible, productImage, productName, productPrice, productOriginalPrice,
  specs, skus, onClose, onConfirm, actionType = 'cart'
}: SkuSelectorProps) {
  const { selectedSpecs, handleSelectSpec, handleConfirm } = useSkuSelectorState(skus, onConfirm);
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  return (
    <SkuSelectorModal
      visible={visible}
      productImage={productImage}
      productName={productName}
      productPrice={productPrice}
      specs={specs}
      selectedSpecs={selectedSpecs}
      colors={colors}
      onClose={onClose}
      onSelectSpec={handleSelectSpec}
      actionType={actionType}
      onConfirm={handleConfirm}
    />
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  container: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    maxHeight: '80%',
  },
  header: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  imageContainer: {
    marginRight: 15,
  },
  imagePlaceholder: {
    width: 80,
    height: 80,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
  },
  info: {
    flex: 1,
  },
  price: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  name: {
    fontSize: 16,
  },
  closeBtn: {
    padding: 5,
  },
  specsContainer: {
    maxHeight: 300,
  },
  specGroup: {
    marginBottom: 20,
  },
  specName: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 10,
  },
  options: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  option: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f5f5f5',
  },
  optionText: {
    fontSize: 14,
  },
  footer: {
    marginTop: 20,
  },
  confirmBtn: {
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  confirmText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
