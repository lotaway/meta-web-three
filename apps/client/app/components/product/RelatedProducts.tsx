import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator, Dimensions } from 'react-native';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { recommendationHooks } from '@/app/lib/api/graphql-hooks';
import type { RecommendationsBySceneQuery } from '@/src/generated/graphql/types';
import { router } from 'expo-router';

const CARD_WIDTH = (Dimensions.get('window').width - 50) / 2;

interface RelatedProductsProps {
  productId: number;
  userId?: number;
  scene?: string;
  colors: any;
}

export function RelatedProducts({ productId, userId = 1, scene = 'similar', colors }: RelatedProductsProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];
  const [recommendations, setRecommendations] = useState<RecommendationsBySceneQuery['recommendationsByScene']>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    recommendationHooks.getRecommendationsByScene(userId, scene, 10)
      .then((data) => {
        if (mounted) setRecommendations(data?.recommendationsByScene ?? []);
      })
      .catch(() => {
        if (mounted) setRecommendations([]);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, [userId, scene]);

  const renderContent = () => {
    if (loading) {
      return (
        <View style={styles.centerContent}>
          <ActivityIndicator size="small" color={themeColors.primary} />
        </View>
      );
    }

    if (recommendations.length === 0) {
      return (
        <View style={styles.centerContent}>
          <Text style={[styles.placeholder, { color: themeColors.fontColorLight }]}>
            暂无相关商品推荐
          </Text>
        </View>
      );
    }

    return (
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.scrollList}>
        {recommendations.map((item, index) => {
          const productIdNum = Number(item?.productId ?? 0)
          return (
            <TouchableOpacity
              key={item?.id ?? index}
              style={[styles.card, { backgroundColor: themeColors.background }]}
              onPress={() => {
                if (item?.id) {
                  recommendationHooks.markClicked(Number(item.id)).catch(() => {})
                }
                router.push({ pathname: '/product/[id]', params: { id: productIdNum } })
              }}
            >
              <View style={[styles.imagePlaceholder, { backgroundColor: themeColors.primary + '15' }]}>
                <Text style={[styles.imagePlaceholderText, { color: themeColors.primary }]}>
                  #{productIdNum}
                </Text>
              </View>
              <View style={styles.cardInfo}>
                <Text numberOfLines={2} style={[styles.cardReason, { color: themeColors.fontColorDark }]}>
                  {item?.reason || `Recommendation #${productIdNum}`}
                </Text>
                <Text style={[styles.cardScore, { color: themeColors.primary }]}>
                  Score {Number(item?.score ?? 0).toFixed(2)}
                </Text>
              </View>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    );
  };

  return (
    <View style={[styles.container, { backgroundColor: '#fff' }]}>
      <View style={styles.header}>
        <View style={styles.headerLine} />
        <Text style={[styles.headerText, { color: themeColors.fontColorDark }]}>相关推荐</Text>
        <View style={styles.headerLine} />
      </View>
      <View style={styles.content}>
        {renderContent()}
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
    width: CARD_WIDTH,
    marginRight: 10,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: '#eee',
  },
  imagePlaceholder: {
    width: CARD_WIDTH,
    height: CARD_WIDTH,
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
