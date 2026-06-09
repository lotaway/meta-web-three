import React from 'react'
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

interface ProfileMenuCellProps {
  icon: string
  title: string
  color: string
  onPress?: () => void
  showBorder?: boolean
  badge?: number
}

export default function ProfileMenuCell({ icon, title, color, onPress, showBorder = true, badge }: ProfileMenuCellProps) {
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  return (
    <TouchableOpacity style={[styles.menuCell, showBorder && styles.bottomBorder]} onPress={onPress}>
      <View style={styles.menuLeft}>
        <IconSymbol name={icon as any} size={20} color={color} />
        <Text style={[styles.menuTitle, { color: colors.fontColorDark }]}>{title}</Text>
        {badge != null && badge > 0 && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>{badge > 99 ? '99+' : badge}</Text>
          </View>
        )}
      </View>
      <IconSymbol name="chevron.right" size={16} color={colors.fontColorLight} />
    </TouchableOpacity>
  )
}

const styles = StyleSheet.create({
  menuCell: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 15,
    paddingHorizontal: 15,
  },
  bottomBorder: {
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  menuLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  menuTitle: {
    fontSize: 15,
    marginLeft: 15,
  },
  badge: {
    backgroundColor: '#FF3B30',
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
    paddingHorizontal: 6,
  },
  badgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },
})
