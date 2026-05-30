import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  Switch,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { addressApi } from '@/api/generated'
import { useAuth } from '@/contexts/AuthContext'
import type { MemberAddress } from '@/src/generated/api/models'

export default function AddressEditScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const { userId } = useAuth()
  const params = useLocalSearchParams()
  const addressId = params.id ? parseInt(params.id as string, 10) : null

  const [name, setName] = useState('')
  const [phone, setPhone] = useState('')
  const [province, setProvince] = useState('')
  const [city, setCity] = useState('')
  const [region, setRegion] = useState('')
  const [detailAddress, setDetailAddress] = useState('')
  const [postCode, setPostCode] = useState('')
  const [isDefault, setIsDefault] = useState(false)
  const [loading, setLoading] = useState(false)
  const [initialLoading, setInitialLoading] = useState(!!addressId)

  useEffect(() => {
    if (addressId) {
      loadAddress()
    }
  }, [addressId])

  const loadAddress = async () => {
    if (!userId) return
    try {
      const response = await addressApi.list({ xUserId: userId })
      const address = response.data?.find((a: MemberAddress) => a.id === addressId)
      if (address) {
        setName(address.name || '')
        setPhone(address.phoneNumber || '')
        setProvince(address.province || '')
        setCity(address.city || '')
        setRegion(address.region || '')
        setDetailAddress(address.detailAddress || '')
        setPostCode(address.postCode || '')
        setIsDefault(address.defaultStatus === true)
      }
    } catch (error) {
      Alert.alert(t('common.error'), t('address.load_failed'))
    } finally {
      setInitialLoading(false)
    }
  }

  const validate = (): boolean => {
    if (!name.trim()) {
      Alert.alert(t('common.error'), t('address.name_required'))
      return false
    }
    if (!phone.trim() || phone.length < 11) {
      Alert.alert(t('common.error'), t('address.phone_required'))
      return false
    }
    if (!province.trim()) {
      Alert.alert(t('common.error'), t('address.province_required'))
      return false
    }
    if (!city.trim()) {
      Alert.alert(t('common.error'), t('address.city_required'))
      return false
    }
    if (!detailAddress.trim()) {
      Alert.alert(t('common.error'), t('address.detail_required'))
      return false
    }
    return true
  }

  const handleSave = async () => {
    if (!validate()) return
    if (!userId) return

    setLoading(true)
    try {
      const addressData: MemberAddress = {
        name,
        phoneNumber: phone,
        province,
        city,
        region,
        detailAddress,
        postCode,
        defaultStatus: isDefault,
      }

      if (addressId) {
        await addressApi.update({ id: addressId, memberAddress: addressData })
      } else {
        await addressApi.add({ memberAddress: addressData })
      }

      Alert.alert(t('common.success'), t('address.save_success'), [
        { text: 'OK', onPress: () => router.back() },
      ])
    } catch (error) {
      Alert.alert(t('common.error'), t('address.save_failed'))
    } finally {
      setLoading(false)
    }
  }

  if (initialLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>
          {addressId ? t('address.edit_title') : t('address.add_title')}
        </Text>
        <TouchableOpacity style={styles.saveBtn} onPress={handleSave} disabled={loading}>
          {loading ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.saveBtnText}>{t('common.save')}</Text>
          )}
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content}>
        <View style={styles.form}>
          <View style={styles.field}>
            <Text style={[styles.label, { color: colors.textSecondary }]}>{t('address.name_label')} *</Text>
            <TextInput
              style={[styles.input, { color: colors.text, borderColor: colors.border }]}
              placeholder={t('address.name_placeholder')}
              placeholderTextColor={colors.textSecondary}
              value={name}
              onChangeText={setName}
            />
          </View>

          <View style={styles.field}>
            <Text style={[styles.label, { color: colors.textSecondary }]}>{t('address.phone_label')} *</Text>
            <TextInput
              style={[styles.input, { color: colors.text, borderColor: colors.border }]}
              placeholder={t('address.phone_placeholder')}
              placeholderTextColor={colors.textSecondary}
              value={phone}
              onChangeText={setPhone}
              keyboardType="phone-pad"
              maxLength={11}
            />
          </View>

          <View style={styles.field}>
            <Text style={[styles.label, { color: colors.textSecondary }]}>{t('address.region_label')} *</Text>
            <View style={styles.regionRow}>
              <TextInput
                style={[styles.regionInput, { color: colors.text, borderColor: colors.border }]}
                placeholder={t('address.province_placeholder')}
                placeholderTextColor={colors.textSecondary}
                value={province}
                onChangeText={setProvince}
              />
              <TextInput
                style={[styles.regionInput, { color: colors.text, borderColor: colors.border }]}
                placeholder={t('address.city_placeholder')}
                placeholderTextColor={colors.textSecondary}
                value={city}
                onChangeText={setCity}
              />
              <TextInput
                style={[styles.regionInput, { color: colors.text, borderColor: colors.border }]}
                placeholder={t('address.region_placeholder')}
                placeholderTextColor={colors.textSecondary}
                value={region}
                onChangeText={setRegion}
              />
            </View>
          </View>

          <View style={styles.field}>
            <Text style={[styles.label, { color: colors.textSecondary }]}>{t('address.detail_label')} *</Text>
            <TextInput
              style={[styles.input, styles.textArea, { color: colors.text, borderColor: colors.border }]}
              placeholder={t('address.detail_placeholder')}
              placeholderTextColor={colors.textSecondary}
              value={detailAddress}
              onChangeText={setDetailAddress}
              multiline
              numberOfLines={3}
              textAlignVertical="top"
            />
          </View>

          <View style={styles.field}>
            <Text style={[styles.label, { color: colors.textSecondary }]}>{t('address.postcode_label')}</Text>
            <TextInput
              style={[styles.input, { color: colors.text, borderColor: colors.border }]}
              placeholder={t('address.postcode_placeholder')}
              placeholderTextColor={colors.textSecondary}
              value={postCode}
              onChangeText={setPostCode}
              keyboardType="number-pad"
              maxLength={6}
            />
          </View>

          <View style={[styles.defaultRow, { borderColor: colors.border }]}>
            <Text style={[styles.defaultLabel, { color: colors.text }]}>{t('address.set_default')}</Text>
            <Switch
              value={isDefault}
              onValueChange={setIsDefault}
              trackColor={{ false: colors.border, true: colors.primary + '60' }}
              thumbColor={isDefault ? colors.primary : '#f4f3f4'}
            />
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  )
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
    marginRight: 50,
  },
  saveBtn: {
    position: 'absolute',
    right: 12,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#333',
    borderRadius: 16,
  },
  saveBtnText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  content: { flex: 1 },
  form: { padding: 16 },
  field: { marginBottom: 20 },
  label: { fontSize: 14, marginBottom: 8 },
  input: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 12,
    fontSize: 16,
  },
  textArea: {
    minHeight: 80,
  },
  regionRow: {
    flexDirection: 'row',
    gap: 10,
  },
  regionInput: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 12,
    fontSize: 14,
  },
  defaultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderTopWidth: 1,
    marginTop: 10,
  },
  defaultLabel: { fontSize: 14 },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
})
