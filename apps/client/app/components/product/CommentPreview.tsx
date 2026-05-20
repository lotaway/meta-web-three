import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { commentApi } from '@/api/generated';

interface CommentPreviewProps {
  productId: number;
  productName: string;
  colors: any;
}

export function CommentPreview({ productId, productName, colors }: CommentPreviewProps) {
  const [comments, setComments] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];

  useEffect(() => {
    loadComments();
  }, [productId]);

  const loadComments = async () => {
    try {
      const response = await commentApi.listByProduct({ productId });
      if (response.data && Array.isArray(response.data)) {
        setComments(response.data.slice(0, 3));
      }
    } catch (error) {
      console.error('Failed to load comments:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return null;
  if (comments.length === 0) {
    return (
      <View style={[styles.commentSection, { backgroundColor: '#fff' }]}>
        <View style={styles.commentHeader}>
          <Text style={[styles.commentTitle, { color: themeColors.fontColorDark }]}>商品评价</Text>
          <Text style={[styles.commentCount, { color: themeColors.fontColorLight }]}>暂无评价</Text>
        </View>
        <View style={styles.commentEmpty}>
          <IconSymbol name="chatbubble" size={40} color={themeColors.fontColorDisabled} />
          <Text style={[styles.commentEmptyText, { color: themeColors.fontColorDisabled }]}>
            暂无用户评价
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.commentSection, { backgroundColor: '#fff' }]}>
      <View style={styles.commentHeader}>
        <Text style={[styles.commentTitle, { color: themeColors.fontColorDark }]}>商品评价</Text>
        <Text style={[styles.commentCount, { color: themeColors.fontColorLight }]}>
          {comments.length} 条评价
        </Text>
      </View>
      {comments.map((comment) => (
        <View key={comment.id} style={styles.commentItem}>
          <View style={styles.commentUserRow}>
            <View style={[styles.commentAvatar, { backgroundColor: themeColors.primary }]}>
              <Text style={styles.commentAvatarText}>
                {comment.memberNickName ? comment.memberNickName.charAt(0) : 'U'}
              </Text>
            </View>
            <View style={styles.commentUserInfo}>
              <Text style={[styles.commentUserName, { color: themeColors.fontColorDark }]}>
                {comment.memberNickName || '匿名用户'}
              </Text>
              <View style={styles.commentStars}>
                {[1, 2, 3, 4, 5].map((star) => (
                  <IconSymbol
                    key={star}
                    name={star <= comment.star ? 'star.fill' : 'star'}
                    size={12}
                    color={star <= comment.star ? '#FFD700' : themeColors.fontColorDisabled}
                  />
                ))}
              </View>
            </View>
          </View>
          <Text style={[styles.commentContent, { color: themeColors.fontColorDark }]} numberOfLines={2}>
            {comment.content}
          </Text>
          {comment.productAttribute && (
            <Text style={[styles.commentAttr, { color: themeColors.fontColorLight }]}>
              {comment.productAttribute}
            </Text>
          )}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  commentSection: {
    marginTop: 15,
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  commentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  commentTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  commentCount: {
    fontSize: 12,
  },
  commentEmpty: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  commentEmptyText: {
    fontSize: 14,
    marginTop: 8,
  },
  commentItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f5f5f5',
  },
  commentUserRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  commentAvatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  commentAvatarText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  commentUserInfo: {
    flex: 1,
    marginLeft: 10,
  },
  commentUserName: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 2,
  },
  commentStars: {
    flexDirection: 'row',
  },
  commentContent: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 4,
  },
  commentAttr: {
    fontSize: 12,
  },
});
