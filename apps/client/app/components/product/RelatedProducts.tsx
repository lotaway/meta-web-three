import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

interface RelatedProductsProps {
  productId: number;
  colors: any;
}

export function RelatedProducts({ productId, colors }: RelatedProductsProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];

  return (
    <View style={[styles.container, { backgroundColor: '#fff' }]}>
      <View style={styles.header}>
        <View style={styles.headerLine} />
        <Text style={[styles.headerText, { color: themeColors.fontColorDark }]}>相关推荐</Text>
        <View style={styles.headerLine} />
      </View>
      <View style={styles.content}>
        <Text style={[styles.placeholder, { color: themeColors.fontColorLight }]}>
          暂无相关商品推荐
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: 15,
    paddingVertical: 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 20,
  },
  headerLine: {
    width: 50,
    height: 1,
    backgroundColor: '#ddd',
    marginHorizontal: 15,
  },
  headerText: {
    fontSize: 16,
    fontWeight: '600',
  },
  content: {
    paddingHorizontal: 20,
    alignItems: 'center',
    paddingVertical: 20,
  },
  placeholder: {
    fontSize: 14,
  },
});
