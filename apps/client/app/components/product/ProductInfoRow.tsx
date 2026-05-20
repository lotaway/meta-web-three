import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { IconSymbol } from '@/components/ui/IconSymbol';

interface ProductInfoRowProps {
  title: string;
  content: string;
  showArrow?: boolean;
  colors: any;
  contentStyle?: any;
  onPress?: () => void;
}

export function ProductInfoRow({ title, content, showArrow, colors, contentStyle, onPress }: ProductInfoRowProps) {
  return (
    <TouchableOpacity style={styles.infoRow} onPress={onPress}>
      <Text style={[styles.rowTitle, { color: colors.fontColorLight }]}>{title}</Text>
      <Text numberOfLines={1} style={[styles.rowContent, { color: colors.fontColorDark }, contentStyle]}>
        {content}
      </Text>
      {showArrow && <IconSymbol name="chevron.right" size={14} color={colors.fontColorLight} />}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f9f9f9',
  },
  rowTitle: {
    width: 80,
    fontSize: 14,
  },
  rowContent: {
    flex: 1,
    fontSize: 14,
  },
});
