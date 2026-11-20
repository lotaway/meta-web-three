import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { useAuth } from '@/contexts/AuthContext';
import { userApi, commissionApi } from '@/api/generated';

interface WalletData {
  integration: number;
  growth: number;
  commissionBalance: number;
}

const QUICK_ACTIONS = [
  { label: '积分充值', icon: 'arrow.up.circle.fill', color: '#FF6B35' },
  { label: '积分提现', icon: 'arrow.down.circle.fill', color: '#4CAF50' },
  { label: '交易记录', icon: 'clock.arrow.circlepath', color: '#2196F3' },
];

type AppColors = typeof Colors.light;
type AppRouter = ReturnType<typeof useRouter>;
type AuthUserId = ReturnType<typeof useAuth>['userId'];
type AuthFields = { isAuthenticated: boolean; userId: AuthUserId };

async function fetchWalletData(isAuthenticated: boolean, userId: AuthUserId): Promise<WalletData | null> {
  if (!isAuthenticated || !userId) return null;
  const [userRes, commRes] = await Promise.all([
    userApi.info({ xUserId: userId }),
    commissionApi.balance({ userId }).catch(() => null),
  ]);
  const user = userRes.data as any;
  return {
    integration: user?.integration ?? 0,
    growth: user?.growth ?? 0,
    commissionBalance: (commRes?.data as any)?.availableBalance ?? 0,
  };
}

async function loadWalletDataFlow(isRefresh: boolean, state: WalletState & AuthFields) {
  if (isRefresh) state.setRefreshing(true);
  else state.setLoading(true);
  try {
    const data = await fetchWalletData(state.isAuthenticated, state.userId);
    if (data) state.setWalletData(data);
  } catch (error) {
    console.error('Failed to load wallet data:', error);
    Alert.alert('Error', 'Failed to load the wallet data.');
  } finally {
    state.setLoading(false);
    state.setRefreshing(false);
  }
}

function useWalletState() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [walletData, setWalletData] = useState<WalletData>({ integration: 0, growth: 0, commissionBalance: 0 });
  return { loading, setLoading, refreshing, setRefreshing, walletData, setWalletData };
}

type WalletState = ReturnType<typeof useWalletState>;

function useWalletController() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { isAuthenticated, userId } = useAuth();
  const state = useWalletState();
  const loadWalletData = (isRefresh = false) => loadWalletDataFlow(isRefresh, { ...state, isAuthenticated, userId });
  useEffect(() => {
    loadWalletData();
  }, [isAuthenticated, userId]);
  return { t, router, colors, isAuthenticated, userId, ...state, loadWalletData };
}

interface WalletViewProps extends WalletState {
  router: AppRouter;
  colors: AppColors;
  isAuthenticated: boolean;
  userId: AuthUserId;
  loadWalletData: (isRefresh?: boolean) => Promise<void>;
}

function WalletHeader({ colors, handleBack }: { colors: AppColors; handleBack: () => void }) {
  return (
    <View style={[styles.header, { borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>我的钱包</Text>
      <View style={styles.headerRight} />
    </View>
  );
}

function LoadingView({ colors, handleBack }: { colors: AppColors; handleBack: () => void }) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <WalletHeader colors={colors} handleBack={handleBack} />
      <View style={styles.center}>
        <ActivityIndicator size="large" color={colors.tint} />
      </View>
    </SafeAreaView>
  );
}

function BalanceCard({ integration, colors }: { integration: number; colors: AppColors }) {
  return (
    <View style={[styles.balanceCard, { backgroundColor: colors.primary }]}>
      <Text style={styles.balanceLabel}>可用积分</Text>
      <Text style={styles.balanceAmount}>{integration}</Text>
      <Text style={styles.balanceUnit}>分</Text>
    </View>
  );
}

function QuickActionItem({ action, colors }: { action: typeof QUICK_ACTIONS[number]; colors: AppColors }) {
  return (
    <TouchableOpacity style={[styles.actionItem, { backgroundColor: colors.card }]}>
      <IconSymbol name={action.icon as any} size={28} color={action.color} />
      <Text style={[styles.actionText, { color: colors.text }]}>{action.label}</Text>
    </TouchableOpacity>
  );
}

function QuickActions({ colors }: { colors: AppColors }) {
  return (
    <View style={styles.quickActions}>
      {QUICK_ACTIONS.map((action) => <QuickActionItem key={action.label} action={action} colors={colors} />)}
    </View>
  );
}

function DetailRow({ label, value, colors }: { label: string; value: string; colors: AppColors }) {
  return (
    <View style={[styles.detailRow, { borderBottomColor: colors.border }]}>
      <Text style={[styles.detailLabel, { color: colors.textSecondary }]}>{label}</Text>
      <Text style={[styles.detailValue, { color: colors.text }]}>{value}</Text>
    </View>
  );
}

function AssetDetails({ walletData, colors }: { walletData: WalletData; colors: AppColors }) {
  return (
    <View style={[styles.detailSection, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>资产明细</Text>
      <DetailRow label="积分" value={String(walletData.integration)} colors={colors} />
      <DetailRow label="成长值" value={String(walletData.growth)} colors={colors} />
      {walletData.commissionBalance > 0 && <DetailRow label="佣金余额" value={`¥${walletData.commissionBalance.toFixed(2)}`} colors={colors} />}
    </View>
  );
}

function WalletView(props: WalletViewProps) {
  const { colors, router, refreshing, walletData, loadWalletData } = props;
  const handleBack = () => router.back();
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <WalletHeader colors={colors} handleBack={handleBack} />
      <ScrollView showsVerticalScrollIndicator={false} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => loadWalletData(true)} />}>
        <BalanceCard integration={walletData.integration} colors={colors} />
        <QuickActions colors={colors} />
        <AssetDetails walletData={walletData} colors={colors} />
        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

export default function WalletScreen() {
  const controller = useWalletController();
  if (controller.loading) {
    return <LoadingView colors={controller.colors} handleBack={() => controller.router.back()} />;
  }
  return <WalletView {...controller} />;
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
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
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  balanceCard: {
    margin: 16,
    padding: 24,
    borderRadius: 16,
    alignItems: 'center',
  },
  balanceLabel: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
    marginBottom: 8,
  },
  balanceAmount: {
    fontSize: 48,
    fontWeight: '700',
    color: '#fff',
  },
  balanceUnit: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },
  quickActions: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    marginBottom: 16,
    gap: 12,
  },
  actionItem: {
    flex: 1,
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
  },
  actionText: {
    fontSize: 13,
    marginTop: 8,
  },
  detailSection: {
    marginHorizontal: 16,
    borderRadius: 12,
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  detailLabel: {
    fontSize: 14,
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '500',
  },
  bottomSpacer: {
    height: 40,
  },
});
