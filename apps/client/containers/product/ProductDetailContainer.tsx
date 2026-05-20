import React, { ReactNode, useEffect, useState } from 'react';
import { toProductDetailVM, ProductDetailVM } from '@/adapters/productAdapter';
import { productApi } from '@/api/generated';

export interface ProductDetailContainerState {
  productDetails: ProductDetailVM | null;
  isPageLoading: boolean;
}

interface ProductDetailContainerProps {
  productId: number | null;
  children: (state: ProductDetailContainerState) => ReactNode;
}

export function ProductDetailContainer({ productId, children }: ProductDetailContainerProps) {
  const [productDetails, setProductDetails] = useState<ProductDetailVM | null>(null);
  const [isPageLoading, setIsPageLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      if (!productId) {
        if (mounted) {
          setProductDetails(null);
          setIsPageLoading(false);
        }
        return;
      }
      try {
        const response = await productApi.getProduct({ id: productId });
        if (mounted) {
          setProductDetails(response.data ? toProductDetailVM(response.data) : null);
        }
      } catch {
        if (mounted) {
          setProductDetails(null);
        }
      } finally {
        if (mounted) {
          setIsPageLoading(false);
        }
      }
    };

    setIsPageLoading(true);
    load();

    return () => {
      mounted = false;
    };
  }, [productId]);

  return <>{children({ productDetails, isPageLoading })}</>;
}
