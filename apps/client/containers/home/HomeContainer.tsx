import React, { ReactNode, useEffect, useState } from 'react'
import { toBrandItem, toProductItem, toAdvertiseItem, MallHomeContent } from '@/adapters/homeAdapter'
import { brandApi, categoryApi, productApi, advertiseApi, API_BASE_URL } from '@/api/generated'
import { ProductCategory } from '@/src/generated/api/models/ProductCategory'

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
          subjectList: [],
        })
      } catch {
        setMallHomeData({
          advertiseList: [],
          brandList: [],
          homeFlashPromotion: null,
          newProductList: [],
          hotProductList: [],
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
