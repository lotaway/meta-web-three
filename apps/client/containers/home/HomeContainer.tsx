import React, { ReactNode, useEffect, useState } from 'react'
import { toBrandItem, toProductItem, toAdvertiseItem, MallHomeContent, ProductItem } from '@/adapters/homeAdapter'
import { brandApi, categoryApi, productApi, advertiseApi, API_BASE_URL } from '@/api/generated'
import { ProductCategory } from '@/src/generated/api/models/ProductCategory'
import { recommendationHooks } from '@/app/lib/api/graphql-hooks'
import type { RecommendationsQuery } from '@/src/generated/graphql/types'

export interface CategoryItem {
  id: number
  name: string
  parentId: number
}

export interface HomeContainerState {
  mallHomeData: MallHomeContent | null
  isDataLoaded: boolean
  fetchProductCategoryList: (parentId: number) => Promise<{ data: CategoryItem[] }>
}

interface HomeContainerProps {
  children: (state: HomeContainerState) => ReactNode
}

function toCategoryItem(pc: ProductCategory): CategoryItem {
  return {
    id: pc.id ?? 0,
    name: pc.name ?? '',
    parentId: pc.parentId ?? 0,
  }
}

function toRecommendationProductItem(data: NonNullable<RecommendationsQuery['recommendations']>[number]): ProductItem {
  return {
    id: Number(data?.productId),
    name: data?.reason || `Recommend #${data?.productId}`,
    pic: '',
    price: 0,
    subTitle: data?.score ? `Score: ${data.score.toFixed(2)}` : undefined,
  }
}

export function HomeContainer({ children }: HomeContainerProps) {
  const [mallHomeData, setMallHomeData] = useState<MallHomeContent | null>(null)
  const [isDataLoaded, setIsDataLoaded] = useState(false)

  const fetchProductCategoryList = async (parentId: number): Promise<{ data: CategoryItem[] }> => {
    const response = await categoryApi.viewChildren({ parentId })
    return { data: response.data?.map(toCategoryItem) ?? [] }
  }

  useEffect(() => {
    const loadMallContent = async () => {
      try {
        const [advertiseRes, brandRes, newRes, hotRes] = await Promise.all([
          advertiseApi.listAvailable(),
          brandApi.list(),
          productApi.listProducts({ keyword: 'new' }),
          productApi.listProducts({ keyword: 'hot' }),
        ])

        let recommendedProductList: ProductItem[] = []
        try {
          const recData = await recommendationHooks.generateRecommendation(1, 'home', 'HYBRID', 10)
          const items = recData?.generateRecommendation?.items ?? []
          recommendedProductList = items.map((item) => ({
            id: Number(item?.skuCode ?? 0),
            name: item?.skuName || `Product #${item?.skuCode}`,
            pic: '',
            price: 0,
            subTitle: item?.reason || undefined,
          }))
        } catch {
          try {
            const fallbackRes = await productApi.listProducts({ keyword: 'hot' })
            recommendedProductList = (fallbackRes.data ?? []).map(toProductItem)
          } catch {
            recommendedProductList = []
          }
        }

        let flashPromotion = null
        try {
          const flashRes = await fetch(`${API_BASE_URL}/promotion-service/flash/current`)
          const flashData = await flashRes.json()
          const promotions: any[] = flashData?.data ?? []
          if (promotions.length > 0) {
            const sessions: any[] = promotions[0]?.sessions ?? []
            const productList = sessions.flatMap((s: any) =>
              (s.products ?? []).map((p: any) => ({
                id: p.productId,
                productId: p.productId,
                name: p.productName ?? '',
                pic: p.productPic ?? '',
                price: p.flashPromotionPrice ?? 0,
              }))
            )
            if (productList.length > 0) {
              const firstSession = sessions[0]
              flashPromotion = {
                productList,
                startTime: firstSession?.startTime,
                endTime: firstSession?.endTime,
              }
            }
          }
        } catch {
          // flash data unavailable, show nothing
        }

        setMallHomeData({
          advertiseList: (advertiseRes.data ?? []).map(toAdvertiseItem),
          brandList: (brandRes.data ?? []).map(toBrandItem),
          homeFlashPromotion: flashPromotion,
          newProductList: (newRes.data ?? []).map(toProductItem),
          hotProductList: (hotRes.data ?? []).map(toProductItem),
          recommendedProductList,
          subjectList: [],
        })
      } catch {
        setMallHomeData({
          advertiseList: [],
          brandList: [],
          homeFlashPromotion: null,
          newProductList: [],
          hotProductList: [],
          recommendedProductList: [],
          subjectList: [],
        })
      } finally {
        setIsDataLoaded(true)
      }
    }

    loadMallContent()
  }, [])

  return <>{children({ mallHomeData, isDataLoaded, fetchProductCategoryList })}</>
}
