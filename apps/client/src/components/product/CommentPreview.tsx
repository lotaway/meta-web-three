import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Alert } from 'react-native';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { commentApi } from '@/api/generated';

interface CommentPreviewProps {
  productId: number;
  productName: string;
  colors: any;
}

async function fetchComments(productId: number, setComments: (comments: any[]) => void, setLoading: (loading: boolean) => void) {
  try {
    const response = await commentApi.listCommentsByProduct({ productId });
    if (response.data && Array.isArray(response.data)) {
      setComments(response.data.slice(0, 3));
    }
  } catch (error) {
    console.error('Failed to load comments:', error);
    Alert.alert('Error', 'Failed to load comments.');
  } finally {
    setLoading(false);
  }
}

function CommentHeader({ countText, themeColors }: { countText: string; themeColors: any }) {
  return (
    <View style={styles.commentHeader}>
      <Text style={[styles.commentTitle, { color: themeColors.fontColorDark }]}>商品评价</Text>
      <Text style={[styles.commentCount, { color: themeColors.fontColorLight }]}>{countText}</Text>
    </View>
  );
}

function EmptyComments({ themeColors }: { themeColors: any }) {
  return (
    <View style={[styles.commentSection, { backgroundColor: '#fff' }]}>
      <CommentHeader countText="暂无评价" themeColors={themeColors} />
      <View style={styles.commentEmpty}>
        <IconSymbol name={'chatbubble' as any} size={40} color={themeColors.fontColorDisabled} />
        <Text style={[styles.commentEmptyText, { color: themeColors.fontColorDisabled }]}>
          暂无用户评价
        </Text>
      </View>
    </View>
  );
}

function CommentStars({ star, themeColors }: { star: number; themeColors: any }) {
  return (
    <View style={styles.commentStars}>
      {[1, 2, 3, 4, 5].map((starValue) => (
        <IconSymbol
          key={starValue}
          name={starValue <= star ? 'star.fill' : 'star'}
          size={12}
          color={starValue <= star ? '#FFD700' : themeColors.fontColorDisabled}
        />
      ))}
    </View>
  );
}

function CommentUserRow({ comment, themeColors }: { comment: any; themeColors: any }) {
  return (
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
        <CommentStars star={comment.star} themeColors={themeColors} />
      </View>
    </View>
  );
}

function CommentItem({ comment, themeColors }: { comment: any; themeColors: any }) {
  return (
    <View style={styles.commentItem}>
      <CommentUserRow comment={comment} themeColors={themeColors} />
      <Text style={[styles.commentContent, { color: themeColors.fontColorDark }]} numberOfLines={2}>
        {comment.content}
      </Text>
      {comment.productAttribute && (
        <Text style={[styles.commentAttr, { color: themeColors.fontColorLight }]}>
          {comment.productAttribute}
        </Text>
      )}
    </View>
  );
}

export function CommentPreview({ productId, productName, colors }: CommentPreviewProps) {
  const [comments, setComments] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];

  useEffect(() => {
    fetchComments(productId, setComments, setLoading);
  }, [productId]);

  if (loading) return null;
  if (comments.length === 0) return <EmptyComments themeColors={themeColors} />;

  return (
    <View style={[styles.commentSection, { backgroundColor: '#fff' }]}>
      <CommentHeader countText={`${comments.length} 条评价`} themeColors={themeColors} />
      {comments.map((comment) => (
        <CommentItem key={comment.id} comment={comment} themeColors={themeColors} />
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
