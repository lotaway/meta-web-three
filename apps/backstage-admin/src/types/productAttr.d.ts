/** Product category attribute info */
export type ProductAttrInfo = {
  /** Product attribute ID */
  attributeId: number
  /** Product attribute category ID */
  attributeCategoryId: number
}

/** Product attribute entity */
export type PmsProductAttribute = {
  /** ID */
  id?: number
  /** Product attribute category ID */
  productAttributeCategoryId?: number
  /** Attribute name */
  name: string
  /** Attribute selection type: 0 -> unique; 1 -> single choice; 2 -> multiple choice */
  selectType?: number
  /** Attribute input method: 0 -> manual; 1 -> select from list */
  inputType?: number
  /** Optional values, comma-separated */
  inputList?: string
  /** Sort field; the highest one can have its own image */
  sort?: number
  /** Category filter style: 0 -> normal; 1 -> color */
  filterType?: number
  /** Search type: 0 -> none; 1 -> keyword; 2 -> range */
  searchType?: number
  /** Whether products with the same attribute are related: 0 -> no; 1 -> yes */
  relatedStatus?: number
  /** Whether manual add is supported: 0 -> no; 1 -> yes */
  handAddStatus?: number
  /** Attribute type: 0 -> specification; 1 -> parameter */
  type?: number
}

/** Product attribute category entity */
export type PmsProductAttributeCategory = {
  /** Attribute ID */
  id?: number
  /** Attribute name */
  name: string
  /** Attribute count */
  attributeCount?: number
  /** Parameter count */
  paramCount?: number
}

/** Product attribute category extension */
export type PmsProductAttributeCategoryExt = PmsProductAttributeCategory & {
  /** Product attribute categories */
  productAttributeList?: PmsProductAttribute[]
}

/** Product attribute value */
export type PmsProductAttributeValue = {
  /** ID */
  id?: number
  /** Product ID */
  productId?: number
  /** Product attribute ID */
  productAttributeId?: number
  /** Manually added spec/parameter value; single for parameter, comma-separated for multiple specs */
  value?: string
}

/** Product specification info */
export interface ProductAttrVo extends Pick<
  PmsProductAttribute,
  'id' | 'name' | 'handAddStatus' | 'inputList'
> {
  /** Selected value */
  values?: string[]
  /** Options */
  options?: string[]
}

/** Product parameter info */
export interface ProductParamVo extends Pick<
  PmsProductAttribute,
  'id' | 'name' | 'inputType' | 'inputList'
> {
  /** Parameter value */
  value?: string
}

/** Product attribute image */
export type ProductAttrPicVo = {
  /** Name */
  name?: string
  /** Image URL */
  pic?: string
}
