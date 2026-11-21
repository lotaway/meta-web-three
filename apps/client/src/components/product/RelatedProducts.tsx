import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator, useWindowDimensions } from 'react-native';
import { useTranslation } from 'react-i18next';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { recommendationHooks } from '@/lib/api/graphql-hooks';
import type { Query } from '@/src/generated/graphql/types';
type RecommendationsBySceneQuery = Pick<Query, 'recommendationsByScene'>;
import { router } from 'expo-router';

interface RelatedProductsProps {
  productId: number;
  userId?: number;
  scene?: string;
  colors: any;
}

function LoadingState({ colors }: { colors: any }) {
  return (
    <View style={styles.centerContent}>
      <ActivityIndicator size="small" color={colors.primary} />
    </View>
  );
}

function EmptyState({ colors }: { colors: any }) {
  const { t } = useTranslation();
  return (
    <View style={styles.centerContent}>
      <Text style={[styles.placeholder, { color: colors.fontColorLight }]}>
        {t('home.product.related.empty')}
      </Text>
    </View>
  );
}

function CardContent({ item, productIdNum, colors }: { item: any; productIdNum: number; colors: any }) {
  const { t } = useTranslation();
  const { width } = useWindowDimensions();
  const cardWidth = (width - 50) / 2;
  return (
    <>
      <View style={[styles.imagePlaceholder, { width: cardWidth, height: cardWidth, backgroundColor: colors.primary + '15' }]}>
        <Text style={[styles.imagePlaceholderText, { color: colors.primary }]}>
          #{productIdNum}
        </Text>
      </View>
      <View style={styles.cardInfo}>
        <Text numberOfLines={2} style={[styles.cardReason, { color: colors.fontColorDark }]}>
          {item?.reason || t('home.product.related.fallback_reason', { id: productIdNum })}
        </Text>
        <Text style={[styles.cardScore, { color: colors.primary }]}>
          {t('home.product.related.score', { score: Number(item?.score ?? 0).toFixed(2) })}
        </Text>
      </View>
    </>
  );
}

function RecommendationCard({ item, colors }: { item: any; colors: any }) {
  const { width } = useWindowDimensions();
  const cardWidth = (width - 50) / 2;
  const productIdNum = Number(item?.productId ?? 0);
  const handlePress = () => {
    if (item?.id) recommendationHooks.markClicked(Number(item.id)).catch(() => {});
    router.push({ pathname: '/product/[id]', params: { id: productIdNum } });
  };
  return (
    <TouchableOpacity
      style={[styles.card, { width: cardWidth, backgroundColor: colors.background }]}
      onPress={handlePress}
    >
      <CardContent item={item} productIdNum={productIdNum} colors={colors} />
    </TouchableOpacity>
  );
}

function RecommendationList({ items, colors }: { items: any[]; colors: any }) {
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.scrollList}>
      {items.map((item: any, index: number) => (
        <RecommendationCard key={item?.id ?? index} item={item} colors={colors} />
      ))}
    </ScrollView>
  );
}

function useRecommendations(userId: number, scene: string) {
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    recommendationHooks.getRecommendationsByScene(userId, scene, 10)
      .then((data) => { if (mounted) setRecommendations((data as any)?.recommendationsByScene ?? []); })
      .catch(() => { if (mounted) setRecommendations([]); })
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, [userId, scene]);

  return { recommendations, loading };
}

function RelatedProductsHeader({ colors, t }: { colors: any; t: (key: string) => string }) {
  return (
    <View style={styles.header}>
      <View style={styles.headerLine} />
      <Text style={[styles.headerText, { color: colors.fontColorDark }]}>
        {t('home.product.related.title')}
      </Text>
      <View style={styles.headerLine} />
    </View>
  );
}

function RelatedContent({ loading, recommendations, colors }: {
  loading: boolean;
  recommendations: any[];
  colors: any;
}) {
  if (loading) return <LoadingState colors={colors} />;
  if (recommendations.length === 0) return <EmptyState colors={colors} />;
  return <RecommendationList items={recommendations} colors={colors} />;
}

export function RelatedProducts({ productId, userId = 1, scene = 'similar', colors }: RelatedProductsProps) {
  const { t } = useTranslation();
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];
  const { recommendations, loading } = useRecommendations(userId, scene);

  return (
    <View style={[styles.container, { backgroundColor: '#fff' }]}>
      <RelatedProductsHeader colors={themeColors} t={t} />
      <View style={styles.content}>
        <RelatedContent loading={loading} recommendations={recommendations} colors={themeColors} />
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
    minHeight: 80,
  },
  centerContent: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 20,
  },
  placeholder: {
    fontSize: 14,
  },
  scrollList: {
    paddingBottom: 10,
  },
  card: {
    width: 162.5,
    marginRight: 10,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: '#eee',
  },
  imagePlaceholder: {
    width: 162.5,
    height: 162.5,
    justifyContent: 'center',
    alignItems: 'center',
  },
  imagePlaceholderText: {
    fontSize: 16,
    fontWeight: '600',
  },
  cardInfo: {
    padding: 8,
  },
  cardReason: {
    fontSize: 13,
    lineHeight: 18,
  },
  cardScore: {
    fontSize: 12,
    fontWeight: '600',
    marginTop: 4,
  },
});
