import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Modal, StyleSheet, ScrollView } from 'react-native';
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

export function SkuSelector({ 
  visible, 
  productImage, 
  productName, 
  productPrice, 
  productOriginalPrice, 
  specs, 
  skus, 
  onClose, 
  onConfirm,
  actionType = 'cart'
}: SkuSelectorProps) {
  const [selectedSpecs, setSelectedSpecs] = useState<Record<string, string>>({});
  const [quantity, setQuantity] = useState(1);
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  const handleSelectSpec = (groupId: string, optionId: string) => {
    setSelectedSpecs(prev => ({ ...prev, [groupId]: optionId }));
  };

  const handleConfirm = () => {
    const sku = findMatchingSKU();
    onConfirm(selectedSpecs, sku, quantity);
  };

  const findMatchingSKU = (): SKUInfo | null => {
    if (skus.length === 0) return null;
    return skus[0];
  };

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <View style={styles.overlay}>
        <View style={[styles.container, { backgroundColor: colors.background }]}>
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

          <ScrollView style={styles.specsContainer}>
            {specs.map(group => (
              <View key={group.id} style={styles.specGroup}>
                <Text style={[styles.specName, { color: colors.fontColorDark }]}>{group.name}</Text>
                <View style={styles.options}>
                  {group.options.map(option => (
                    <TouchableOpacity
                      key={option.id}
                      style={[
                        styles.option,
                        selectedSpecs[group.id] === option.id && { backgroundColor: colors.primary }
                      ]}
                      onPress={() => handleSelectSpec(group.id, option.id)}
                    >
                      <Text style={[
                        styles.optionText,
                        { color: selectedSpecs[group.id] === option.id ? '#fff' : colors.fontColorDark }
                      ]}>
                        {option.name}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            ))}
          </ScrollView>

          <View style={styles.footer}>
            <TouchableOpacity 
              style={[styles.confirmBtn, { backgroundColor: actionType === 'buy' ? colors.primary : '#ff9500' }]}
              onPress={handleConfirm}
            >
              <Text style={styles.confirmText}>{actionType === 'buy' ? '立即购买' : '加入购物车'}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
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
