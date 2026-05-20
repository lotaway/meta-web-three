import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
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

export default function WalletScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { isAuthenticated, userId } = useAuth();

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [walletData, setWalletData] = useState<WalletData>({
    integration: 0,
    growth: 0,
    commissionBalance: 0,
  });

  const loadWalletData = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    try {
      if (isAuthenticated && userId) {
        const [userRes, commRes] = await Promise.all([
          userApi.info({ xUserId: userId }),
          commissionApi.balance({ userId }).catch(() => null),
        ]);
        const user = userRes.data as any;
        setWalletData({
          integration: user?.integration ?? 0,
          growth: user?.growth ?? 0,
          commissionBalance: (commRes?.data as any)?.availableBalance ?? 0,
        });
      }
    } catch (error) {
      console.error('Failed to load wallet data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadWalletData();
  }, [isAuthenticated, userId]);

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
            <IconSymbol name="chevron.left" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>我的钱包</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.center}>
          <ActivityIndicator size="large" color={colors.tint} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>我的钱包</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={() => loadWalletData(true)} />
        }
      >
        <View style={[styles.balanceCard, { backgroundColor: colors.primary }]}>
          <Text style={styles.balanceLabel}>可用积分</Text>
          <Text style={styles.balanceAmount}>{walletData.integration}</Text>
          <Text style={styles.balanceUnit}>分</Text>
        </View>

        <View style={styles.quickActions}>
          <TouchableOpacity style={[styles.actionItem, { backgroundColor: colors.card }]}>
            <IconSymbol name="arrow.up.circle.fill" size={28} color="#FF6B35" />
            <Text style={[styles.actionText, { color: colors.text }]}>积分充值</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.actionItem, { backgroundColor: colors.card }]}>
            <IconSymbol name="arrow.down.circle.fill" size={28} color="#4CAF50" />
            <Text style={[styles.actionText, { color: colors.text }]}>积分提现</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.actionItem, { backgroundColor: colors.card }]}>
            <IconSymbol name="clock.arrow.circlepath" size={28} color="#2196F3" />
            <Text style={[styles.actionText, { color: colors.text }]}>交易记录</Text>
          </TouchableOpacity>
        </View>

        <View style={[styles.detailSection, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>资产明细</Text>
          <View style={[styles.detailRow, { borderBottomColor: colors.border }]}>
            <Text style={[styles.detailLabel, { color: colors.textSecondary }]}>积分</Text>
            <Text style={[styles.detailValue, { color: colors.text }]}>{walletData.integration}</Text>
          </View>
          <View style={[styles.detailRow, { borderBottomColor: colors.border }]}>
            <Text style={[styles.detailLabel, { color: colors.textSecondary }]}>成长值</Text>
            <Text style={[styles.detailValue, { color: colors.text }]}>{walletData.growth}</Text>
          </View>
          {walletData.commissionBalance > 0 && (
            <View style={[styles.detailRow, { borderBottomColor: colors.border }]}>
              <Text style={[styles.detailLabel, { color: colors.textSecondary }]}>佣金余额</Text>
              <Text style={[styles.detailValue, { color: colors.text }]}>
                ¥{walletData.commissionBalance.toFixed(2)}
              </Text>
            </View>
          )}
        </View>

        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
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
