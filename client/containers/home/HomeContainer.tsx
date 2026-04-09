import React, { ReactNode, useEffect, useState } from 'react';
import { toBrandItem, toProductItem, MallHomeContent } from '@/adapters/homeAdapter';
import { brandApi, categoryApi, productApi } from '@/api/generated';
import { ProductCategory } from '@/src/generated/api/models/ProductCategory';

export interface CategoryItem {
  id: number;
  name: string;
  parentId: number;
}

export interface HomeContainerState {
  mallHomeData: MallHomeContent | null;
  isDataLoaded: boolean;
  fetchProductCategoryList: (parentId: number) => Promise<{ data: CategoryItem[] }>;
}

interface HomeContainerProps {
  children: (state: HomeContainerState) => ReactNode;
}

function toCategoryItem(pc: ProductCategory): CategoryItem {
  return {
    id: pc.id ?? 0,
    name: pc.name ?? '',
    parentId: pc.parentId ?? 0,
  };
}

export function HomeContainer({ children }: HomeContainerProps) {
  const [mallHomeData, setMallHomeData] = useState<MallHomeContent | null>(null);
  const [isDataLoaded, setIsDataLoaded] = useState(false);

  const fetchProductCategoryList = async (parentId: number): Promise<{ data: CategoryItem[] }> => {
    const response = await categoryApi.viewChildren({ parentId });
    return { data: response.data?.map(toCategoryItem) ?? [] };
  };

  useEffect(() => {
    const loadMallContent = async () => {
      try {
        const [brandRes, newRes, hotRes] = await Promise.all([
          brandApi.list(),
          productApi.listProducts({ keyword: 'new' }),
          productApi.listProducts({ keyword: 'hot' }),
        ]);

        setMallHomeData({
          advertiseList: [],
          brandList: (brandRes.data ?? []).map(toBrandItem),
          homeFlashPromotion: null,
          newProductList: (newRes.data ?? []).map(toProductItem),
          hotProductList: (hotRes.data ?? []).map(toProductItem),
          subjectList: [],
        });
      } catch {
        setMallHomeData({
          advertiseList: [],
          brandList: [],
          homeFlashPromotion: null,
          newProductList: [],
          hotProductList: [],
          subjectList: [],
        });
      } finally {
        setIsDataLoaded(true);
      }
    };

    loadMallContent();
  }, []);

  return <>{children({ mallHomeData, isDataLoaded, fetchProductCategoryList })}</>;
}
