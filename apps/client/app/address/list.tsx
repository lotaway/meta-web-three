import React, { useState, useCallback } from 'react'
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native'
import { useRouter } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { useFocusEffect } from '@react-navigation/native'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { addressApi } from '@/api/generated'
import { useAuth } from '@/contexts/AuthContext'
import type { MemberAddress } from '@/src/generated/api/models'

type Translate = ReturnType<typeof useTranslation>['t']
type AppRouter = ReturnType<typeof useRouter>
type AuthContextValue = ReturnType<typeof useAuth>
type UserId = NonNullable<AuthContextValue['userId']>
type SetAddresses = React.Dispatch<React.SetStateAction<MemberAddress[]>>
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>

interface AddressInfoProps {
  item: MemberAddress
  colors: typeof Colors.light
  t: Translate
}

interface CardProps extends AddressInfoProps {
  onEdit: (address: MemberAddress) => void
  onDelete: (id: number) => void
  onSetDefault: (address: MemberAddress) => void
}

interface ListViewProps {
  t: Translate
  colors: typeof Colors.light
  addresses: MemberAddress[]
  loading: boolean
  onAdd: () => void
  onEdit: (address: MemberAddress) => void
  onDelete: (id: number) => void
  onSetDefault: (address: MemberAddress) => void
}

async function fetchUserAddresses(
  userId: UserId,
  setAddresses: SetAddresses
): Promise<void> {
  const response = await addressApi.list3({ xUserId: userId })
  if (response.data) {
    setAddresses(response.data)
  }
}

async function runAddressLoad(
  userId: AuthContextValue['userId'],
  setAddresses: SetAddresses,
  setLoading: SetBoolean
): Promise<void> {
  if (!userId) return
  setLoading(true)
  try {
    await fetchUserAddresses(userId, setAddresses)
  } catch (error) {
    console.error('Failed to load addresses:', error)
    Alert.alert('Error', 'Failed to load addresses.')
  } finally {
    setLoading(false)
  }
}

function useAddressLoader(userId: AuthContextValue['userId']) {
  const [addresses, setAddresses] = useState<MemberAddress[]>([])
  const [loading, setLoading] = useState(true)
  const loadAddresses = useCallback(
    () => runAddressLoad(userId, setAddresses, setLoading),
    [userId]
  )
  useFocusEffect(useCallback(() => { loadAddresses() }, [loadAddresses]))
  return { addresses, loading, loadAddresses }
}

async function removeAddress(id: number, reload: () => void, t: Translate): Promise<void> {
  try {
    await addressApi.remove({ id })
    reload()
  } catch (error) {
    Alert.alert(t('common.error'), t('address.delete_failed'))
  }
}

async function updateDefaultAddress(
  address: MemberAddress,
  reload: () => void,
  t: Translate
): Promise<void> {
  try {
    await addressApi.update7({
      id: address.id!,
      memberAddress: { ...address, defaultStatus: true },
    })
    reload()
  } catch (error) {
    Alert.alert(t('common.error'), t('address.update_failed'))
  }
}

function useListAuth() {
  const { t } = useTranslation()
  const router = useRouter()
  const colors = Colors[useColorScheme() ?? 'light']
  const auth = useAuth()
  return { t, router, colors, auth }
}

function useListActions(router: AppRouter, t: Translate, reload: () => void) {
  const onAdd = () => router.push('/address/edit')
  const onEdit = (address: MemberAddress) =>
    router.push({ pathname: '/address/edit', params: { id: address.id } })
  const onDelete = (id: number) => {
    Alert.alert(t('address.delete_confirm_title'), t('address.delete_confirm_message'), [
      { text: t('common.cancel'), style: 'cancel' },
      {
        text: t('address.delete'),
        style: 'destructive',
        onPress: () => removeAddress(id, reload, t),
      },
    ])
  }
  const onSetDefault = (address: MemberAddress) => updateDefaultAddress(address, reload, t)
  return { onAdd, onEdit, onDelete, onSetDefault }
}

function AddressHeader({ colors, t, onBack }: ListViewProps & { onBack: () => void }) {
  return (
    <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>{t('address.title')}</Text>
      <View style={styles.headerRight} />
    </View>
  )
}

function AddressCardHeader({ item, colors, t }: AddressInfoProps) {
  const isDefault = item.defaultStatus === true
  return (
    <View style={styles.addressHeader}>
      <View style={styles.userInfo}>
        <Text style={[styles.userName, { color: colors.text }]}>{item.name}</Text>
        <Text style={[styles.userPhone, { color: colors.textSecondary }]}>{item.phoneNumber}</Text>
      </View>
      {isDefault && (
        <View style={[styles.defaultTag, { backgroundColor: colors.primary }]}>
          <Text style={styles.defaultTagText}>{t('address.default')}</Text>
        </View>
      )}
    </View>
  )
}

function ItemActionButton({
  icon,
  color,
  label,
  onPress,
}: {
  icon: 'pencil' | 'trash'
  color: string
  label: string
  onPress: () => void
}) {
  return (
    <TouchableOpacity style={styles.actionBtn} onPress={onPress}>
      <IconSymbol name={icon} size={18} color={color} />
      <Text style={[styles.actionText, { color }]}>{label}</Text>
    </TouchableOpacity>
  )
}

function DefaultButton({ colors, t, item, onSetDefault }: Pick<CardProps, 'colors' | 't' | 'item' | 'onSetDefault'>) {
  return (
    <TouchableOpacity style={styles.defaultBtn} onPress={() => onSetDefault(item)}>
      <Text style={[styles.defaultBtnText, { color: colors.primary }]}>
        {t('address.set_default')}
      </Text>
    </TouchableOpacity>
  )
}

function AddressActions({ item, colors, t, onEdit, onDelete, onSetDefault }: CardProps) {
  const isDefault = item.defaultStatus === true
  return (
    <View style={styles.addressFooter}>
      {!isDefault && <DefaultButton colors={colors} t={t} item={item} onSetDefault={onSetDefault} />}
      <View style={styles.actions}>
        <ItemActionButton icon="pencil" color={colors.primary} label={t('address.edit')} onPress={() => onEdit(item)} />
        <ItemActionButton icon="trash" color="#FF3B30" label={t('address.delete')} onPress={() => onDelete(item.id!)} />
      </View>
    </View>
  )
}

function AddressCard({ item, colors, t, onEdit, onDelete, onSetDefault }: CardProps) {
  return (
    <TouchableOpacity
      style={[styles.addressCard, { backgroundColor: colors.card, borderColor: colors.border }]}
      onPress={() => onEdit(item)}
    >
      <AddressCardHeader item={item} colors={colors} t={t} />
      <Text style={[styles.addressText, { color: colors.text }]}>
        {item.province}{item.city}{item.region}{item.detailAddress}
      </Text>
      <AddressActions item={item} colors={colors} t={t} onEdit={onEdit} onDelete={onDelete} onSetDefault={onSetDefault} />
    </TouchableOpacity>
  )
}

function LoadingState({ colors }: { colors: typeof Colors.light }) {
  return (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color={colors.primary} />
    </View>
  )
}

function EmptyState({ colors, t, onAdd }: ListViewProps) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="mappin.and.ellipse" size={60} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('address.no_addresses')}</Text>
      <TouchableOpacity style={[styles.addBtn, { backgroundColor: colors.primary }]} onPress={onAdd}>
        <Text style={styles.addBtnText}>{t('address.add')}</Text>
      </TouchableOpacity>
    </View>
  )
}

function AddressListRows({
  addresses,
  colors,
  t,
  onEdit,
  onDelete,
  onSetDefault,
}: ListViewProps) {
  return (
    <FlatList
      data={addresses}
      keyExtractor={(item) => String(item.id)}
      renderItem={({ item }) => (
        <AddressCard item={item} colors={colors} t={t} onEdit={onEdit} onDelete={onDelete} onSetDefault={onSetDefault} />
      )}
      contentContainerStyle={styles.listContent}
    />
  )
}

function BottomAddButton({ colors, t, onAdd }: ListViewProps) {
  return (
    <TouchableOpacity
      style={[styles.bottomAddBtn, { backgroundColor: colors.primary }]}
      onPress={onAdd}
    >
      <IconSymbol name="plus" size={20} color="#fff" />
      <Text style={styles.bottomAddBtnText}>{t('address.add_new')}</Text>
    </TouchableOpacity>
  )
}

function AddressListContent(props: ListViewProps) {
  return (
    <>
      <AddressListRows {...props} />
      <BottomAddButton {...props} />
    </>
  )
}

function AddressListView(props: ListViewProps & { onBack: () => void }) {
  const { loading, addresses } = props
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: props.colors.background }]}>
      <AddressHeader {...props} />
      {loading ? (
        <LoadingState colors={props.colors} />
      ) : addresses.length === 0 ? (
        <EmptyState {...props} />
      ) : (
        <AddressListContent {...props} />
      )}
    </SafeAreaView>
  )
}

export default function AddressListScreen() {
  const { t, router, colors, auth } = useListAuth()
  const { addresses, loading, loadAddresses } = useAddressLoader(auth.userId)
  const actions = useListActions(router, t, loadAddresses)
  const onBack = () => router.back()
  return <AddressListView t={t} colors={colors} addresses={addresses} loading={loading} {...actions} onBack={onBack} />
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
    marginRight: 40,
  },
  headerRight: { width: 40 },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyText: {
    marginTop: 12,
    fontSize: 14,
    marginBottom: 24,
  },
  addBtn: {
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 24,
  },
  addBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  listContent: {
    padding: 12,
    paddingBottom: 100,
  },
  addressCard: {
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 12,
  },
  addressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  userInfo: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  userName: { fontSize: 16, fontWeight: '600' },
  userPhone: { fontSize: 14 },
  defaultTag: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  defaultTagText: { color: '#fff', fontSize: 12 },
  addressText: { fontSize: 14, lineHeight: 20, marginBottom: 12 },
  addressFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
    paddingTop: 12,
  },
  defaultBtn: { padding: 4 },
  defaultBtnText: { fontSize: 13 },
  actions: { flexDirection: 'row', gap: 16 },
  actionBtn: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  actionText: { fontSize: 13 },
  bottomAddBtn: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 14,
    borderRadius: 24,
    gap: 8,
  },
  bottomAddBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
})
