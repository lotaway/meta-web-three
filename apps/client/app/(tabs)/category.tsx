import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { CategoryContainer } from '@/containers/category/CategoryContainer';

type AppColors = typeof Colors.light;

function navigateToProductList(subCategoryId: number, subCategoryName: string) {
  router.push({
    pathname: '/category/[id]',
    params: { id: subCategoryId, name: subCategoryName },
  });
}

function PrimaryCategoryItem({ category, selectedId, colors, onSelect }: {
  category: any; selectedId: number | null; colors: AppColors; onSelect: (id: number) => void
}) {
  const selected = selectedId === category.id;
  return (
    <TouchableOpacity
      key={category.id}
      style={[styles.fItem, selected && { backgroundColor: colors.background }]}
      onPress={() => onSelect(category.id)}
    >
      {selected && <View style={[styles.activeLine, { backgroundColor: colors.primary }]} />}
      <Text style={[styles.fItemText, { color: selected ? colors.primary : colors.fontColorBase }]}>
        {category.name}
      </Text>
    </TouchableOpacity>
  );
}

function SecondaryCategoryItem({ subCategory, colors }: { subCategory: any; colors: AppColors }) {
  return (
    <TouchableOpacity
      key={subCategory.id}
      style={styles.sItem}
      onPress={() => navigateToProductList(subCategory.id, subCategory.name)}
    >
      <Image source={{ uri: subCategory.icon || 'https://via.placeholder.com/70' }} style={styles.sImage} />
      <Text style={[styles.sItemText, { color: colors.fontColorBase }]}>{subCategory.name}</Text>
    </TouchableOpacity>
  );
}

function PrimaryCategoryList({ primaryCategories, selectedMainCategoryId, colors, onMainCategorySelect }: {
  primaryCategories: any[]; selectedMainCategoryId: number | null; colors: AppColors; onMainCategorySelect: (id: number) => void
}) {
  return (
    <ScrollView style={styles.leftAside} showsVerticalScrollIndicator={false}>
      {primaryCategories.map((category) => (
        <PrimaryCategoryItem key={category.id} category={category} selectedId={selectedMainCategoryId} colors={colors} onSelect={onMainCategorySelect} />
      ))}
    </ScrollView>
  );
}

function SecondaryCategoryList({ secondaryCategories, colors }: { secondaryCategories: any[]; colors: AppColors }) {
  return (
    <ScrollView style={styles.rightAside} showsVerticalScrollIndicator={false}>
      <View style={styles.sList}>
        {secondaryCategories.map((subCategory) => (
          <SecondaryCategoryItem key={subCategory.id} subCategory={subCategory} colors={colors} />
        ))}
      </View>
    </ScrollView>
  );
}

function CategoryContent({ data, colors }: { data: any; colors: AppColors }) {
  const { primaryCategories, secondaryCategories, selectedMainCategoryId, onMainCategorySelect } = data;
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.content}>
        <PrimaryCategoryList primaryCategories={primaryCategories} selectedMainCategoryId={selectedMainCategoryId} colors={colors} onMainCategorySelect={onMainCategorySelect} />
        <SecondaryCategoryList secondaryCategories={secondaryCategories} colors={colors} />
      </View>
    </SafeAreaView>
  );
}

export default function CategoryScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  return (
    <CategoryContainer>
      {(data) => <CategoryContent data={data} colors={colors} />}
    </CategoryContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    flexDirection: 'row',
  },
  leftAside: {
    width: 100,
    backgroundColor: '#fff',
  },
  fItem: {
    width: '100%',
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  activeLine: {
    position: 'absolute',
    left: 0,
    top: '30%',
    height: '40%',
    width: 4,
    borderRadius: 2,
  },
  fItemText: {
    fontSize: 14,
  },
  rightAside: {
    flex: 1,
    paddingLeft: 10,
  },
  sList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    backgroundColor: '#fff',
    marginTop: 10,
    paddingVertical: 10,
    borderRadius: 8,
  },
  sItem: {
    width: '33.33%',
    alignItems: 'center',
    paddingBottom: 20,
  },
  sImage: {
    width: 70,
    height: 70,
    marginBottom: 8,
  },
  sItemText: {
    fontSize: 12,
  },
});
