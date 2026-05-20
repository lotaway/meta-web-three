import React, { ReactNode, useEffect, useState } from 'react';
import { categoryApi } from '@/api/generated';
import { ProductCategory } from '@/src/generated/api/models/ProductCategory';

export interface CategoryVM {
  id: number;
  name: string;
  icon?: string;
  parentId: number;
}

export interface CategoryContainerState {
  primaryCategories: CategoryVM[];
  secondaryCategories: CategoryVM[];
  selectedMainCategoryId: number | null;
  onMainCategorySelect: (categoryId: number) => void;
}

interface CategoryContainerProps {
  rootCategoryId?: number;
  children: (state: CategoryContainerState) => ReactNode;
}

function toCategoryVM(pc: ProductCategory): CategoryVM {
  return {
    id: pc.id ?? 0,
    name: pc.name ?? '',
    icon: pc.icon ?? undefined,
    parentId: pc.parentId ?? 0,
  };
}

export function CategoryContainer({ rootCategoryId = 0, children }: CategoryContainerProps) {
  const [primaryCategories, setPrimaryCategories] = useState<CategoryVM[]>([]);
  const [secondaryCategories, setSecondaryCategories] = useState<CategoryVM[]>([]);
  const [selectedMainCategoryId, setSelectedMainCategoryId] = useState<number | null>(null);

  const fetchChildren = async (parentId: number) => {
    const response = await categoryApi.viewChildren({ parentId });
    return response.data?.map(toCategoryVM) ?? [];
  };

  const onMainCategorySelect = (categoryId: number) => {
    setSelectedMainCategoryId(categoryId);
    fetchChildren(categoryId).then(setSecondaryCategories).catch(() => setSecondaryCategories([]));
  };

  useEffect(() => {
    const init = async () => {
      try {
        const root = await fetchChildren(rootCategoryId);
        setPrimaryCategories(root);
        if (root.length > 0) {
          const firstId = root[0].id;
          setSelectedMainCategoryId(firstId);
          const second = await fetchChildren(firstId);
          setSecondaryCategories(second);
        }
      } catch {
        setPrimaryCategories([]);
        setSecondaryCategories([]);
        setSelectedMainCategoryId(null);
      }
    };
    init();
  }, [rootCategoryId]);

  return <>{children({ primaryCategories, secondaryCategories, selectedMainCategoryId, onMainCategorySelect })}</>;
}
