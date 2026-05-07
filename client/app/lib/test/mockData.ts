export interface Product {
  id: number
  name: string
  pic: string
  price: number
  originalPrice?: number
  stock: number
  sale: number
  description?: string
  detail?: string
  categoryId?: number
  brandId?: number
  status: number
  createTime: string
  updateTime?: string
}

export interface ProductCategory {
  id: number
  name: string
  parentId?: number
  level: number
  icon?: string
  sort: number
  children?: ProductCategory[]
}

export interface Brand {
  id: number
  name: string
  logo?: string
  productCount: number
  status: number
}

export interface Order {
  id: number
  orderSn: string
  memberId: number
  totalAmount: number
  payAmount: number
  freightAmount?: number
  status: number
  payType?: number
  payTime?: string
  deliveryCompany?: string
  deliverySn?: string
  receiverName: string
  receiverPhone: string
  receiverPostCode?: string
  receiverProvince: string
  receiverCity: string
  receiverRegion: string
  receiverDetailAddress: string
  note?: string
  orderItems: OrderItem[]
  createTime: string
  updateTime?: string
}

export interface OrderItem {
  id: number
  orderId: number
  productId: number
  productPic: string
  productName: string
  productPrice: number
  productQuantity: number
  totalPrice: number
}

export interface CartItem {
  id: number
  memberId: number
  productId: number
  quantity: number
  product?: Product
}

export interface User {
  id: number
  username: string
  nickname?: string
  phone?: string
  email?: string
  avatar?: string
  status: number
  createTime: string
}

export interface Address {
  id: number
  memberId: number
  name: string
  phone: string
  province: string
  city: string
  region: string
  detailAddress: string
  isDefault: number
  createTime: string
}

export interface HomeContent {
  newProductList: Product[]
  hotProductList: Product[]
  brandList: Brand[]
  advertiseList: Advertise[]
  subjectList: Subject[]
  homeFlashPromotion?: FlashPromotion
}

export interface Advertise {
  id: number
  name: string
  pic: string
  link?: string
}

export interface Subject {
  id: number
  title: string
  pic: string
  categoryId?: number
}

export interface FlashPromotion {
  startTime: string
  endTime: string
  flashProductList: Product[]
}

export function generateProduct(overrides: Partial<Product> = {}): Product {
  const id = Math.floor(Math.random() * 10000)
  return {
    id,
    name: `商品名称 ${id}`,
    pic: `https://picsum.photos/200/200?random=${id}`,
    price: Math.floor(Math.random() * 1000) + 100,
    originalPrice: Math.floor(Math.random() * 2000) + 500,
    stock: Math.floor(Math.random() * 100),
    sale: Math.floor(Math.random() * 1000),
    description: '商品描述',
    categoryId: Math.floor(Math.random() * 10) + 1,
    brandId: Math.floor(Math.random() * 5) + 1,
    status: 1,
    createTime: new Date().toISOString(),
    ...overrides,
  }
}

export function generateProductList(count: number = 10): Product[] {
  return Array.from({ length: count }, () => generateProduct())
}

export function generateCategory(overrides: Partial<ProductCategory> = {}): ProductCategory {
  const id = Math.floor(Math.random() * 1000)
  return {
    id,
    name: `分类名称 ${id}`,
    parentId: 0,
    level: 1,
    sort: 0,
    children: [],
    ...overrides,
  }
}

export function generateBrand(overrides: Partial<Brand> = {}): Brand {
  const id = Math.floor(Math.random() * 100)
  return {
    id,
    name: `品牌名称 ${id}`,
    logo: `https://picsum.photos/100/100?random=${id}`,
    productCount: Math.floor(Math.random() * 100),
    status: 1,
    ...overrides,
  }
}

export function generateOrder(overrides: Partial<Order> = {}): Order {
  const id = Math.floor(Math.random() * 10000)
  const totalAmount = Math.floor(Math.random() * 10000) + 100
  return {
    id,
    orderSn: `ORDER${Date.now()}${id}`,
    memberId: 1,
    totalAmount,
    payAmount: totalAmount,
    status: 1,
    receiverName: '收货人',
    receiverPhone: '13800138000',
    receiverProvince: '广东省',
    receiverCity: '深圳市',
    receiverRegion: '南山区',
    receiverDetailAddress: '详细地址',
    orderItems: [],
    createTime: new Date().toISOString(),
    ...overrides,
  }
}

export function generateOrderWithItems(itemCount: number = 3): Order {
  const order = generateOrder()
  order.orderItems = Array.from({ length: itemCount }, () => ({
    id: Math.floor(Math.random() * 10000),
    orderId: order.id,
    productId: Math.floor(Math.random() * 1000),
    productPic: `https://picsum.photos/100/100?random=${Math.random()}`,
    productName: `商品 ${Math.floor(Math.random() * 1000)}`,
    productPrice: Math.floor(Math.random() * 500) + 50,
    productQuantity: Math.floor(Math.random() * 5) + 1,
    totalPrice: Math.floor(Math.random() * 1000) + 100,
  }))
  return order
}

export function generateCartItem(overrides: Partial<CartItem> = {}): CartItem {
  return {
    id: Math.floor(Math.random() * 10000),
    memberId: 1,
    productId: Math.floor(Math.random() * 1000),
    quantity: Math.floor(Math.random() * 5) + 1,
    product: generateProduct(),
    ...overrides,
  }
}

export function generateUser(overrides: Partial<User> = {}): User {
  return {
    id: 1,
    username: 'testuser',
    nickname: '测试用户',
    phone: '13800138000',
    email: 'test@example.com',
    avatar: 'https://picsum.photos/100/100',
    status: 1,
    createTime: new Date().toISOString(),
    ...overrides,
  }
}

export function generateAddress(overrides: Partial<Address> = {}): Address {
  return {
    id: Math.floor(Math.random() * 1000),
    memberId: 1,
    name: '收货人',
    phone: '13800138000',
    province: '广东省',
    city: '深圳市',
    region: '南山区',
    detailAddress: '详细地址',
    isDefault: 0,
    createTime: new Date().toISOString(),
    ...overrides,
  }
}

export function generateHomeContent(): HomeContent {
  return {
    newProductList: generateProductList(10),
    hotProductList: generateProductList(10),
    brandList: Array.from({ length: 10 }, () => generateBrand()),
    advertiseList: Array.from({ length: 5 }, (_, i) => ({
      id: i + 1,
      name: `广告 ${i + 1}`,
      pic: `https://picsum.photos/750/300?random=${i}`,
    })),
    subjectList: Array.from({ length: 4 }, (_, i) => ({
      id: i + 1,
      title: `专题 ${i + 1}`,
      pic: `https://picsum.photos/400/200?random=${i + 10}`,
    })),
    homeFlashPromotion: {
      startTime: new Date().toISOString(),
      endTime: new Date(Date.now() + 3600000 * 4).toISOString(),
      flashProductList: generateProductList(10),
    },
  }
}

export function generatePageData<T>(data: T[], page: number = 1, pageSize: number = 10) {
  const total = data.length
  const start = (page - 1) * pageSize
  const end = start + pageSize

  return {
    list: data.slice(start, end),
    pageNum: page,
    pageSize,
    total,
    totalPages: Math.ceil(total / pageSize),
  }
}