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

type Translate = ReturnType<typeof useTranslation>['t']
type AppRouter = ReturnType<typeof useRouter>
type EditParams = ReturnType<typeof useLocalSearchParams>
type AuthContextValue = ReturnType<typeof useAuth>
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetText = React.Dispatch<React.SetStateAction<string>>
type ValidationRule = () => string | null

interface AddressFormValues {
  name: string
  phone: string
  province: string
  city: string
  region: string
  detailAddress: string
  postCode: string
  isDefault: boolean
}

interface AddressSetters {
  setName: SetText
  setPhone: SetText
  setProvince: SetText
  setCity: SetText
  setRegion: SetText
  setDetailAddress: SetText
  setPostCode: SetText
  setIsDefault: SetBoolean
}

interface EditState extends AddressFormValues, AddressSetters {
  loading: boolean
  setLoading: SetBoolean
  initialLoading: boolean
  setInitialLoading: SetBoolean
}

interface EditViewProps extends AddressFormValues, AddressSetters {
  t: Translate
  colors: typeof Colors.light
  addressId: number | null
  loading: boolean
  onSave: () => void
  onBack: () => void
}

function parseAddressId(params: EditParams): number | null {
  return params.id ? parseInt(params.id as string, 10) : null
}

function firstValidationError(rules: ValidationRule[]): string | null {
  for (const rule of rules) {
    const error = rule()
    if (error) {
      return error
    }
  }
  return null
}

function getAddressValidationError(values: AddressFormValues, t: Translate): string | null {
  const rules: ValidationRule[] = [
    () => (!values.name.trim() ? t('address.name_required') : null),
    () => (!values.phone.trim() || values.phone.length < 11 ? t('address.phone_required') : null),
    () => (!values.province.trim() ? t('address.province_required') : null),
    () => (!values.city.trim() ? t('address.city_required') : null),
    () => (!values.detailAddress.trim() ? t('address.detail_required') : null),
  ]
  return firstValidationError(rules)
}

function buildAddressData(values: AddressFormValues): MemberAddress {
  return {
    name: values.name,
    phoneNumber: values.phone,
    province: values.province,
    city: values.city,
    region: values.region,
    detailAddress: values.detailAddress,
    postCode: values.postCode,
    defaultStatus: values.isDefault,
  }
}

function applyAddressToForm(address: MemberAddress, setters: AddressSetters) {
  setters.setName(address.name || '')
  setters.setPhone(address.phoneNumber || '')
  setters.setProvince(address.province || '')
  setters.setCity(address.city || '')
  setters.setRegion(address.region || '')
  setters.setDetailAddress(address.detailAddress || '')
  setters.setPostCode(address.postCode || '')
  setters.setIsDefault(address.defaultStatus === true)
}

async function loadAddressById(
  addressId: number,
  userId: AuthContextValue['userId'],
  setters: AddressSetters,
  setInitialLoading: SetBoolean,
  t: Translate
): Promise<void> {
  if (!userId) return
  try {
    const response = await addressApi.list3({ xUserId: userId })
    const address = response.data?.find((item: MemberAddress) => item.id === addressId)
    if (address) {
      applyAddressToForm(address, setters)
    }
  } catch (error) {
    Alert.alert(t('common.error'), t('address.load_failed'))
  } finally {
    setInitialLoading(false)
  }
}

function showSaveSuccess(t: Translate, router: AppRouter) {
  Alert.alert(t('common.success'), t('address.save_success'), [
    { text: 'OK', onPress: () => router.back() },
  ])
}

async function saveAddress(values: AddressFormValues, addressId: number | null, setLoading: SetBoolean, t: Translate, router: AppRouter): Promise<void> {
  setLoading(true)
  try {
    const addressData = buildAddressData(values)
    if (addressId) {
      await addressApi.update7({ id: addressId, memberAddress: addressData })
    } else {
      await addressApi.add({ memberAddress: addressData })
    }
    showSaveSuccess(t, router)
  } catch (error) {
    Alert.alert(t('common.error'), t('address.save_failed'))
  } finally {
    setLoading(false)
  }
}

function useEditAuth() {
  const { t } = useTranslation()
  const router = useRouter()
  const colors = Colors[useColorScheme() ?? 'light']
  const auth = useAuth()
  const params = useLocalSearchParams()
  const addressId = parseAddressId(params)
  return { t, router, colors, auth, addressId }
}

function useEditState(addressId: number | null): EditState {
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
  return {
    name, setName, phone, setPhone, province, setProvince, city, setCity,
    region, setRegion, detailAddress, setDetailAddress, postCode, setPostCode,
    isDefault, setIsDefault, loading, setLoading, initialLoading, setInitialLoading,
  }
}

function useAddressLoadEffect(
  state: EditState,
  auth: AuthContextValue,
  addressId: number | null,
  t: Translate
) {
  const setters: AddressSetters = {
    setName: state.setName, setPhone: state.setPhone, setProvince: state.setProvince,
    setCity: state.setCity, setRegion: state.setRegion, setDetailAddress: state.setDetailAddress,
    setPostCode: state.setPostCode, setIsDefault: state.setIsDefault,
  }
  useEffect(() => {
    if (addressId) {
      loadAddressById(addressId, auth.userId, setters, state.setInitialLoading, t)
    }
  }, [addressId])
}

function useEditActions(state: EditState, auth: AuthContextValue, addressId: number | null, t: Translate, router: AppRouter) {
  useAddressLoadEffect(state, auth, addressId, t)
  const onSave = () => {
    const error = getAddressValidationError(state, t)
    if (error) {
      Alert.alert(t('common.error'), error)
      return
    }
    if (!auth.userId) return
    return saveAddress(state, addressId, state.setLoading, t, router)
  }
  const onBack = () => router.back()
  return { onSave, onBack }
}

function EditHeader({ colors, t, addressId, loading, onSave, onBack }: EditViewProps) {
  return (
    <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>
        {addressId ? t('address.edit_title') : t('address.add_title')}
      </Text>
      <TouchableOpacity style={styles.saveBtn} onPress={onSave} disabled={loading}>
        {loading ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Text style={styles.saveBtnText}>{t('common.save')}</Text>
        )}
      </TouchableOpacity>
    </View>
  )
}

function NameField({ colors, name, setName, t }: EditViewProps) {
  return (
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
  )
}

function PhoneField({ colors, phone, setPhone, t }: EditViewProps) {
  return (
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
  )
}

function RegionInput({ colors, placeholder, value, onChangeText }: {
  colors: typeof Colors.light; placeholder: string; value: string; onChangeText: (text: string) => void
}) {
  return (
    <TextInput
      style={[styles.regionInput, { color: colors.text, borderColor: colors.border }]}
      placeholder={placeholder}
      placeholderTextColor={colors.textSecondary}
      value={value}
      onChangeText={onChangeText}
    />
  )
}

function RegionFields({ colors, province, setProvince, city, setCity, region, setRegion, t }: EditViewProps) {
  return (
    <View style={styles.field}>
      <Text style={[styles.label, { color: colors.textSecondary }]}>{t('address.region_label')} *</Text>
      <View style={styles.regionRow}>
        <RegionInput colors={colors} placeholder={t('address.province_placeholder')} value={province} onChangeText={setProvince} />
        <RegionInput colors={colors} placeholder={t('address.city_placeholder')} value={city} onChangeText={setCity} />
        <RegionInput colors={colors} placeholder={t('address.region_placeholder')} value={region} onChangeText={setRegion} />
      </View>
    </View>
  )
}

function DetailField({ colors, detailAddress, setDetailAddress, t }: EditViewProps) {
  return (
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
  )
}

function PostCodeField({ colors, postCode, setPostCode, t }: EditViewProps) {
  return (
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
  )
}

function DefaultSwitch({ colors, isDefault, setIsDefault, t }: EditViewProps) {
  return (
    <View style={[styles.defaultRow, { borderColor: colors.border }]}>
      <Text style={[styles.defaultLabel, { color: colors.text }]}>{t('address.set_default')}</Text>
      <Switch
        value={isDefault}
        onValueChange={setIsDefault}
        trackColor={{ false: colors.border, true: colors.primary + '60' }}
        thumbColor={isDefault ? colors.primary : '#f4f3f4'}
      />
    </View>
  )
}

function EditForm(props: EditViewProps) {
  return (
    <View style={styles.form}>
      <NameField {...props} />
      <PhoneField {...props} />
      <RegionFields {...props} />
      <DetailField {...props} />
      <PostCodeField {...props} />
      <DefaultSwitch {...props} />
    </View>
  )
}

function EditView(props: EditViewProps) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: props.colors.background }]}>
      <EditHeader {...props} />
      <ScrollView style={styles.content}>
        <EditForm {...props} />
      </ScrollView>
    </SafeAreaView>
  )
}

export default function AddressEditScreen() {
  const { t, router, colors, auth, addressId } = useEditAuth()
  const state = useEditState(addressId)
  const actions = useEditActions(state, auth, addressId, t, router)
  if (state.initialLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      </SafeAreaView>
    )
  }
  return <EditView t={t} colors={colors} addressId={addressId} {...state} {...actions} />
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
