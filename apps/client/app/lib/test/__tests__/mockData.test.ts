import {
  generateProduct,
  generateProductList,
  generateCategory,
  generateBrand,
  generateOrder,
  generateOrderWithItems,
  generateCartItem,
  generateUser,
  generateAddress,
  generateHomeContent,
  generatePageData,
} from '../test/mockData'

describe('mockData', () => {
  describe('generateProduct', () => {
    it('should generate product with default values', () => {
      const product = generateProduct()

      expect(product).toMatchObject({
        id: expect.any(Number),
        name: expect.any(String),
        pic: expect.any(String),
        price: expect.any(Number),
        stock: expect.any(Number),
        sale: expect.any(Number),
        status: 1,
        createTime: expect.any(String),
      })
    })

    it('should override default values', () => {
      const product = generateProduct({ name: 'Custom Product', price: 999 })

      expect(product.name).toBe('Custom Product')
      expect(product.price).toBe(999)
    })
  })

  describe('generateProductList', () => {
    it('should generate list with correct count', () => {
      const list = generateProductList(5)
      expect(list).toHaveLength(5)
    })

    it('should generate default 10 products', () => {
      const list = generateProductList()
      expect(list).toHaveLength(10)
    })
  })

  describe('generateCategory', () => {
    it('should generate category with default values', () => {
      const category = generateCategory()

      expect(category).toMatchObject({
        id: expect.any(Number),
        name: expect.any(String),
        parentId: 0,
        level: 1,
        sort: 0,
        children: [],
      })
    })
  })

  describe('generateBrand', () => {
    it('should generate brand with default values', () => {
      const brand = generateBrand()

      expect(brand).toMatchObject({
        id: expect.any(Number),
        name: expect.any(String),
        logo: expect.any(String),
        productCount: expect.any(Number),
        status: 1,
      })
    })
  })

  describe('generateOrder', () => {
    it('should generate order with default values', () => {
      const order = generateOrder()

      expect(order).toMatchObject({
        id: expect.any(Number),
        orderSn: expect.stringContaining('ORDER'),
        memberId: 1,
        totalAmount: expect.any(Number),
        payAmount: expect.any(Number),
        status: 1,
        receiverName: '收货人',
        receiverPhone: '13800138000',
        orderItems: [],
      })
    })

    it('should generate order with items', () => {
      const order = generateOrderWithItems(5)

      expect(order.orderItems).toHaveLength(5)
      order.orderItems.forEach((item) => {
        expect(item).toMatchObject({
          id: expect.any(Number),
          orderId: order.id,
          productId: expect.any(Number),
          productName: expect.any(String),
          productPrice: expect.any(Number),
          productQuantity: expect.any(Number),
          totalPrice: expect.any(Number),
        })
      })
    })
  })

  describe('generateCartItem', () => {
    it('should generate cart item with embedded product', () => {
      const item = generateCartItem()

      expect(item).toMatchObject({
        id: expect.any(Number),
        memberId: 1,
        productId: expect.any(Number),
        quantity: expect.any(Number),
        product: expect.objectContaining({
          id: expect.any(Number),
          name: expect.any(String),
        }),
      })
    })
  })

  describe('generateUser', () => {
    it('should generate user with default values', () => {
      const user = generateUser()

      expect(user).toMatchObject({
        id: 1,
        username: 'testuser',
        nickname: '测试用户',
        phone: '13800138000',
        email: 'test@example.com',
        status: 1,
        createTime: expect.any(String),
      })
    })
  })

  describe('generateAddress', () => {
    it('should generate address with default values', () => {
      const address = generateAddress()

      expect(address).toMatchObject({
        id: expect.any(Number),
        memberId: 1,
        name: '收货人',
        phone: '13800138000',
        province: '广东省',
        city: '深圳市',
        region: '南山区',
        detailAddress: '详细地址',
        isDefault: 0,
      })
    })
  })

  describe('generateHomeContent', () => {
    it('should generate complete home content', () => {
      const content = generateHomeContent()

      expect(content).toMatchObject({
        newProductList: expect.any(Array),
        hotProductList: expect.any(Array),
        brandList: expect.any(Array),
        advertiseList: expect.any(Array),
        subjectList: expect.any(Array),
        homeFlashPromotion: expect.objectContaining({
          startTime: expect.any(String),
          endTime: expect.any(String),
          flashProductList: expect.any(Array),
        }),
      })

      expect(content.newProductList).toHaveLength(10)
      expect(content.hotProductList).toHaveLength(10)
      expect(content.brandList).toHaveLength(10)
      expect(content.advertiseList).toHaveLength(5)
      expect(content.subjectList).toHaveLength(4)
    })
  })

  describe('generatePageData', () => {
    it('should paginate data correctly', () => {
      const data = Array.from({ length: 25 }, (_, i) => i)
      const page1 = generatePageData(data, 1, 10)
      const page2 = generatePageData(data, 2, 10)
      const page3 = generatePageData(data, 3, 10)

      expect(page1.list).toHaveLength(10)
      expect(page1.pageNum).toBe(1)
      expect(page1.total).toBe(25)
      expect(page1.totalPages).toBe(3)

      expect(page2.list).toHaveLength(10)
      expect(page2.list[0]).toBe(10)

      expect(page3.list).toHaveLength(5)
    })
  })
})