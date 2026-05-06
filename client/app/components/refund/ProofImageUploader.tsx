import React from 'react'
import { View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'

interface ProofImageUploaderProps {
  proofImages: string[]
  onRemoveImage: (index: number) => void
  onPickImage: () => void
}

export function ProofImageUploader({ proofImages, onRemoveImage, onPickImage }: ProofImageUploaderProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.proof_title')}</Text>
      <View style={styles.proofGrid}>
        {proofImages.map((uri, index) => (
          <View key={index} style={styles.proofImageContainer}>
            <Image source={{ uri }} style={styles.proofImage} />
            <TouchableOpacity
              style={styles.removeImageBtn}
              onPress={() => onRemoveImage(index)}
            >
              <IconSymbol name="xmark.circle.fill" size={24} color="#FF3B30" />
            </TouchableOpacity>
          </View>
        ))}
        {proofImages.length < 5 && (
          <TouchableOpacity style={[styles.addImageBtn, { borderColor: colors.border }]} onPress={onPickImage}>
            <IconSymbol name="plus" size={32} color={colors.textSecondary} />
            <Text style={[styles.addImageText, { color: colors.textSecondary }]}>{t('refund.add_image')}</Text>
          </TouchableOpacity>
        )}
      </View>
      <Text style={[styles.proofHint, { color: colors.textSecondary }]}>
        {t('refund.proof_hint')}
      </Text>
    </View>
  )
}

const styles = StyleSheet.create({
  section: {
    padding: 16,
    marginTop: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  proofGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  proofImageContainer: {
    width: 90,
    height: 90,
    borderRadius: 8,
    overflow: 'hidden',
    position: 'relative',
  },
  proofImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  removeImageBtn: {
    position: 'absolute',
    top: -4,
    right: -4,
  },
  addImageBtn: {
    width: 90,
    height: 90,
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
  proofHint: {
    fontSize: 12,
    marginTop: 8,
  },
})
