import React from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TouchableOpacity,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { AdvertiseItem, BrandItem, FlashPromotion, ProductItem } from '@/adapters/homeAdapter';
import { HomeContainer } from '@/containers/home/HomeContainer';

const CATE_ITEMS = [
  { nameKey: 'home.categories.topic', icon: 'star.fill' },
  { nameKey: 'home.categories.talk', icon: 'bubble.left.fill' },
  { nameKey: 'home.categories.selected', icon: 'checkmark.seal.fill' },
  { nameKey: 'home.categories.discount', icon: 'tag.fill' },
];

type AppColors = typeof Colors.light;
type Translate = ReturnType<typeof useTranslation>['t'];
type ProductPress = (id: number) => void;

function navigateToProductDetail(productId: number) {
  router.push({ pathname: '/product/[id]', params: { id: productId } });
}

function HomeHeader({ colors }: { colors: AppColors }) {
  const { t } = useTranslation();
  return (
    <View style={[styles.header, { backgroundColor: colors.background }]}>
      <TouchableOpacity testID="search-input" style={styles.searchBox} onPress={() => router.push('/search')}>
        <IconSymbol name="magnifyingglass" size={20} color={colors.fontColorLight} />
        <Text style={[styles.searchText, { color: colors.fontColorLight }]}>{t('common.search_placeholder')}</Text>
      </TouchableOpacity>
      <View style={styles.headerIcons}>
        <TouchableOpacity style={styles.headerIcon} onPress={() => router.push('/scanner')}>
          <IconSymbol name="qrcode.viewfinder" size={24} color={colors.fontColorDark} />
        </TouchableOpacity>
        <TouchableOpacity style={styles.headerIcon} onPress={() => router.push('/notifications')}>
          <IconSymbol name="bell" size={24} color={colors.fontColorDark} />
        </TouchableOpacity>
      </View>
    </View>
  );
}

function BannerSection({ advertiseList }: { advertiseList: AdvertiseItem[] }) {
  const { width } = useWindowDimensions();
  return (
    <ScrollView
      horizontal
      pagingEnabled
      showsHorizontalScrollIndicator={false}
      style={styles.bannerContainer}
    >
      {advertiseList.map((item) => (
        <TouchableOpacity key={item.id} activeOpacity={0.9}>
          <Image source={{ uri: item.pic }} style={[styles.bannerImage, { width: width - 30 }]} />
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
}

function CategorySection({ colors }: { colors: AppColors }) {
  const { t } = useTranslation();
  return (
    <View style={styles.cateSection}>
      {CATE_ITEMS.map((item, index) => (
        <TouchableOpacity key={index} style={styles.cateItem}>
          <View style={[styles.cateIconBox, { backgroundColor: colors.primary + '20' }]}>
            <IconSymbol name={item.icon as any} size={32} color={colors.primary} />
          </View>
          <Text style={[styles.cateName, { color: colors.fontColorDark }]}>{t(item.nameKey)}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

function BrandItemView({ item, colors }: { item: BrandItem; colors: AppColors }) {
  const { t } = useTranslation();
  const { width } = useWindowDimensions();
  return (
    <TouchableOpacity style={[styles.brandItem, { width: (width - 40) / 2 }]} onPress={() => router.push({ pathname: '/brand/[id]', params: { id: item.id } })}>
      <Image source={{ uri: item.logo }} style={styles.brandLogo} resizeMode="contain" />
      <Text style={[styles.brandName, { color: colors.fontColorDark }]}>{item.name}</Text>
      <Text style={[styles.brandCount, { color: colors.fontColorLight }]}>
        {t('home.product.item_count', { count: item.productCount })}
      </Text>
    </TouchableOpacity>
  );
}

function BrandSection({ brandList, colors }: { brandList: BrandItem[]; colors: AppColors }) {
  const { t } = useTranslation();
  return (
    <>
      <SectionHeader
        title={t('home.sections.brand_direct')}
        subtitle={t('home.sections.brand_direct_subtitle')}
        colors={colors}
        onMorePress={() => router.push('/brand')}
      />
      <View style={styles.brandGrid}>
        {brandList.slice(0, 4).map((item) => <BrandItemView key={item.id} item={item} colors={colors} />)}
      </View>
    </>
  );
}

function FlashSection({ promotion, colors, onProductPress }: { promotion: FlashPromotion; colors: AppColors; onProductPress: ProductPress }) {
  const { t } = useTranslation();
  return (
    <>
      <SectionHeader title={t('home.sections.flash_sale')} subtitle={t('home.sections.flash_sale_subtitle')} colors={colors} />
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.flashScroll}>
        {promotion.productList.map((item) => (
          <TouchableOpacity key={item.id} style={styles.flashItem} onPress={() => onProductPress(item.id)}>
            <Image source={{ uri: item.pic }} style={styles.flashImage} />
            <Text numberOfLines={1} style={[styles.productTitle, { color: colors.fontColorDark }]}>
              {item.name}
            </Text>
            <Text style={[styles.price, { color: colors.primary }]}>￥{item.price}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </>
  );
}

function ProductGridItem({ item, colors, onProductPress }: { item: ProductItem; colors: AppColors; onProductPress: ProductPress }) {
  const { width } = useWindowDimensions();
  const cardSize = (width - 30) / 2;
  return (
    <TouchableOpacity style={[styles.productItem, { width: cardSize }]} onPress={() => onProductPress(item.id)}>
      <Image source={{ uri: item.pic }} style={[styles.productImage, { width: cardSize, height: cardSize }]} />
      <View style={styles.productInfo}>
        <Text numberOfLines={1} style={[styles.productTitle, { color: colors.fontColorDark }]}>{item.name}</Text>
        <Text numberOfLines={1} style={[styles.productSubtitle, { color: colors.fontColorLight }]}>{item.subTitle}</Text>
        <Text style={[styles.price, { color: colors.primary }]}>￥{item.price}</Text>
      </View>
    </TouchableOpacity>
  );
}

function ProductGridSection({ title, subtitle, productList, colors, onProductPress }: { title: string; subtitle: string; productList: ProductItem[]; colors: AppColors; onProductPress: ProductPress }) {
  return (
    <>
      <SectionHeader title={title} subtitle={subtitle} colors={colors} />
      <View style={styles.productGrid}>
        {productList.map((item) => <ProductGridItem key={item.id} item={item} colors={colors} onProductPress={onProductPress} />)}
      </View>
    </>
  );
}

function SectionHeader({ title, subtitle, colors, onMorePress }: { title: string; subtitle: string; colors: AppColors; onMorePress?: () => void }) {
  return (
    <View style={styles.sectionHeader}>
      <View style={styles.sectionTitleBox}>
        <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>{title}</Text>
        <Text style={[styles.sectionSubtitle, { color: colors.fontColorLight }]}>{subtitle}</Text>
      </View>
      {onMorePress && (
        <TouchableOpacity onPress={onMorePress} style={styles.moreBtn}>
          <Text style={[styles.moreText, { color: colors.fontColorLight }]}>查看更多</Text>
          <IconSymbol name="chevron.right" size={14} color={colors.fontColorLight} />
        </TouchableOpacity>
      )}
    </View>
  );
}

function HomeLoadingView({ t }: { t: Translate }) {
  return (
    <View testID="home-page" style={[styles.container, styles.center]}>
      <Text>{t('common.loading')}</Text>
    </View>
  );
}

interface HomeContentProps {
  mallHomeData: any;
  isDataLoaded: boolean;
  onRecommendProductPress: ProductPress;
  colors: AppColors;
  t: Translate;
}

function HomeReadyView(props: HomeContentProps) {
  const { mallHomeData, colors, t, onRecommendProductPress } = props;
  return (
    <SafeAreaView testID="home-page" style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView stickyHeaderIndices={[0]} showsVerticalScrollIndicator={false}>
        <HomeHeader colors={colors} />
        <BannerSection advertiseList={mallHomeData?.advertiseList || []} />
        <CategorySection colors={colors} />
        {mallHomeData?.brandList && <BrandSection brandList={mallHomeData.brandList} colors={colors} />}
        {mallHomeData?.homeFlashPromotion && <FlashSection promotion={mallHomeData.homeFlashPromotion} colors={colors} onProductPress={navigateToProductDetail} />}
        <ProductGridSection title={t('home.sections.new_products')} subtitle={t('home.sections.new_products_subtitle')} productList={mallHomeData?.newProductList || []} colors={colors} onProductPress={navigateToProductDetail} />
        <ProductGridSection title={t('home.sections.hot_recommend')} subtitle={t('home.sections.hot_recommend_subtitle')} productList={mallHomeData?.recommendedProductList || []} colors={colors} onProductPress={onRecommendProductPress} />
        <View style={styles.footerSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

function HomeContent(props: HomeContentProps) {
  const { t, isDataLoaded } = props;
  if (!isDataLoaded) return <HomeLoadingView t={t} />;
  return <HomeReadyView {...props} />;
}

export default function HomeScreen() {
  const { t } = useTranslation();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  return (
    <HomeContainer>
      {(content) => <HomeContent {...content} colors={colors} t={t} />}
    </HomeContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  center: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 15,
    paddingVertical: 10,
    zIndex: 10,
  },
  searchBox: {
    flex: 1,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#eee',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 15,
  },
  searchText: {
    marginLeft: 8,
    fontSize: 14,
  },
  headerIcons: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 10,
  },
  headerIcon: {
    marginLeft: 15,
  },
  bannerContainer: {
    height: 180,
  },
  bannerImage: {
    width: 345,
    height: 160,
    marginHorizontal: 15,
    borderRadius: 8,
    marginTop: 10,
  },
  cateSection: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 20,
    backgroundColor: '#fff',
    marginTop: 10,
  },
  cateItem: {
    alignItems: 'center',
  },
  cateIconBox: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  cateName: {
    fontSize: 13,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 15,
    paddingVertical: 20,
    marginTop: 10,
    backgroundColor: '#fff',
  },
  moreBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
    paddingLeft: 12,
  },
  moreText: {
    fontSize: 13,
    marginRight: 2,
  },
  sectionTitleBox: {
    flex: 1,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  sectionSubtitle: {
    fontSize: 12,
    marginTop: 4,
  },
  brandGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 10,
    backgroundColor: '#fff',
  },
  brandItem: {
    width: 167.5,
    marginHorizontal: 5,
    padding: 10,
    marginBottom: 10,
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    alignItems: 'center',
  },
  brandLogo: {
    width: 80,
    height: 40,
    marginBottom: 10,
  },
  brandName: {
    fontSize: 14,
    fontWeight: '500',
  },
  brandCount: {
    fontSize: 11,
    marginTop: 2,
  },
  flashScroll: {
    paddingHorizontal: 10,
    backgroundColor: '#fff',
    paddingBottom: 20,
  },
  flashItem: {
    width: 140,
    marginHorizontal: 5,
  },
  flashImage: {
    width: 140,
    height: 140,
    borderRadius: 8,
    marginBottom: 8,
  },
  productGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 10,
    backgroundColor: '#fff',
  },
  productItem: {
    width: 172.5,
    marginHorizontal: 5,
    marginBottom: 20,
  },
  productImage: {
    width: 172.5,
    height: 172.5,
    borderRadius: 8,
  },
  productInfo: {
    paddingVertical: 8,
  },
  productTitle: {
    fontSize: 14,
    fontWeight: '500',
  },
  productSubtitle: {
    fontSize: 12,
    marginTop: 2,
  },
  price: {
    fontSize: 15,
    fontWeight: 'bold',
    marginTop: 4,
  },
  footerSpacer: {
    height: 100,
  },
});
