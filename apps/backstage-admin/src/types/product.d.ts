import type { PageParam } from './common'
import type { PmsProductAttributeValue } from './productAttr'
import type { PmsSkuStock } from './skuStock'

export type PmsProduct = {
  /** ID */
  id?: number
  /** Brand ID */
  brandId?: number
  /** Product category ID */
  productCategoryId?: number
  /** Shipping template ID */
  feightTemplateId?: number
  /** Attribute category ID */
  productAttributeCategoryId?: number
  /** Name */
  name: string
  /** Image */
  pic?: string
  /** SKU number */
  productSn: string
  /** Deletion status: 0 -> not deleted; 1 -> deleted */
  deleteStatus?: number
  /** Listing status: 0 -> delisted; 1 -> listed */
  publishStatus?: number
  /** New-product status: 0 -> not new; 1 -> new */
  newStatus?: number
  /** Recommendation status: 0 -> not recommended; 1 -> recommended */
  recommandStatus?: number
  /** Review status: 0 -> unreviewed; 1 -> approved */
  verifyStatus?: number
  /** Sort order */
  sort?: number
  /** Sales volume */
  sale?: number
  /** Price */
  price?: number
  /** Promotion price */
  promotionPrice?: number
  /** Bonus growth value */
  giftGrowth?: number
  /** Bonus points */
  giftPoint?: number
  /** Max usable points */
  usePointLimit?: number
  /** Subtitle */
  subTitle?: string
  /** Market price */
  originalPrice?: number
  /** Stock */
  stock?: number
  /** Stock warning threshold */
  lowStock?: number
  /** Unit */
  unit?: string
  /** Product weight, in grams by default */
  weight?: number
  /** Whether pre-release product: 0 -> no; 1 -> yes */
  previewStatus?: number
  /** Comma-separated services: 1 -> worry-free return; 2 -> fast refund; 3 -> free shipping */
  serviceIds?: string
  /** Keyword */
  keywords?: string
  /** Remark */
  note?: string
  /** Gallery images, up to 5 including product images, comma-separated */
  albumPics?: string
  /** Detail title */
  detailTitle?: string
  /** Promotion start time */
  promotionStartTime?: string
  /** Promotion end time */
  promotionEndTime?: string
  /** Activity purchase limit */
  promotionPerLimit?: number
  /** Promotion type: 0 -> none (original price); 1 -> promo price; 2 -> member price; 3 -> tiered price; 4 -> full-reduction; 5 -> flash sale */
  promotionType?: number
  /** Brand name */
  brandName?: string
  /** Product category name */
  productCategoryName?: string
  /** Product description */
  description?: string
  /** Detail description */
  detailDesc?: string
  /** Product detail page content */
  detailHtml?: string
  /** Mobile detail page */
  detailMobileHtml?: string
}

/** Product list query params */
export type ProductQueryParam = PageParam & {
  /** Listing status */
  publishStatus?: number
  /** Review status */
  verifyStatus?: number
  /** Product SKU number */
  productSn?: string
  /** Product category number */
  productCategoryId?: number
  /** Product brand number */
  brandId?: number
}

/** Product create/update request params */
export type PmsProductParam = PmsProduct & {
  /** Product tiered price settings */
  productLadderList?: PmsProductLadder[]
  /** Product full-reduction price settings */
  productFullReductionList?: PmsProductFullReduction[]
  /** Product member price settings */
  memberPriceList?: PmsMemberPrice[]
  /** Product SKU stock info */
  skuStockList?: PmsSkuStock[]
  /** Product params and custom spec attributes */
  productAttributeValueList?: PmsProductAttributeValue[]
  /** Subject-product relation  */
  subjectProductRelationList?: CmsSubjectProductRelation[]
  /** Featured zone-product relation  */
  prefrenceAreaProductRelationList?: CmsPrefrenceAreaProductRelation[]
  /** Parent ID of the product category */
  cateParentId?: number
  /** Flash sale product count */
  flashPromotionCount: number
  /** Flash sale activity ID */
  flashPromotionId: number
  /** Flash sale product price */
  flashPromotionPrice: number
  /** Flash sale sort price */
  flashPromotionSort: number
}

/** Product tiered price */
export type PmsProductLadder = {
  /** ID */
  id?: number
  /** Product ID */
  productId?: number
  /** Required product quantity */
  count: number
  /** Discount */
  discount?: number
  /** Discounted price */
  price?: number
}

/** Product full-reduction price settings */
export type PmsProductFullReduction = {
  /** ID */
  id?: number
  /** Product ID */
  productId?: number
  /** Required amount */
  fullPrice?: number
  /** Reduction amount */
  reducePrice?: number
}

/** Product member price */
export type PmsMemberPrice = {
  /** ID */
  id?: number
  /** Product ID */
  productId?: number
  /** Member level ID */
  memberLevelId?: number
  /** Member price */
  memberPrice?: number
  /** Member level name */
  memberLevelName?: string
}

/** Subject-product relation */
export type CmsSubjectProductRelation = {
  /** ID */
  id?: number
  /** Subject ID */
  subjectId?: number
  /** Product ID */
  productId?: number
}

/** Featured zone-product relation */
export type CmsPrefrenceAreaProductRelation = {
  /** ID */
  id?: number
  /** Featured zone ID */
  prefrenceAreaId?: number
  /** Product ID */
  productId?: number
}
