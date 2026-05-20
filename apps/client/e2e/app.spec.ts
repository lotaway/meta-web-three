import { describe, it, beforeAll, afterAll, expect } from '@jest/globals'
import { init, cleanup, reload, device } from 'detox'
import {
  HomePage,
  CategoryPage,
  ProductDetailPage,
  CartPage,
  CheckoutPage,
  OrderPage,
  UserPage,
  SearchPage,
  LoginPage,
} from '../pages'

const homePage = new HomePage()
const categoryPage = new CategoryPage()
const productDetailPage = new ProductDetailPage()
const cartPage = new CartPage()
const checkoutPage = new CheckoutPage()
const orderPage = new OrderPage()
const userPage = new UserPage()
const searchPage = new SearchPage()
const loginPage = new LoginPage()

describe('首页功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
  })

  afterAll(async () => {
    await cleanup()
  })

  it('应该能够打开应用并显示首页', async () => {
    await homePage.waitForLoad()
    const isVisible = await homePage.visible
    expect(isVisible).toBe(true)
  })

  it('应该能够滚动商品列表', async () => {
    await homePage.scrollToBottom()
  })

  it('应该能够点击搜索栏跳转搜索页', async () => {
    await homePage.tapSearchBar()
    await searchPage.waitForLoad()
    const isVisible = await searchPage.visible
    expect(isVisible).toBe(true)
    await device.back()
  })

  it('应该能够切换到分类页', async () => {
    await homePage.tapCategoryTab()
    await categoryPage.waitForLoad()
    const isVisible = await categoryPage.visible
    expect(isVisible).toBe(true)
  })

  it('应该能够切换到品牌页', async () => {
    await homePage.tapBrandTab()
  })
})

describe('分类功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
    await homePage.tapCategoryTab()
  })

  it('应该能够加载分类列表', async () => {
    await categoryPage.waitForLoad()
    const names = await categoryPage.getCategoryNames()
    expect(names.length).toBeGreaterThan(0)
  })

  it('应该能够点击分类进入商品列表', async () => {
    await categoryPage.tapCategory(1)
  })

  it('应该能够滚动分类列表', async () => {
    await categoryPage.scrollCategoryList('up')
  })
})

describe('商品详情功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
    await homePage.waitForLoad()
  })

  it('应该能够打开商品详情页', async () => {
    await homePage.tapProduct(1)
    await productDetailPage.waitForLoad()
    const isVisible = await productDetailPage.visible
    expect(isVisible).toBe(true)
  })

  it('应该能够获取商品名称', async () => {
    const name = await productDetailPage.getProductName()
    expect(name).toBeTruthy()
  })

  it('应该能够获取商品价格', async () => {
    const price = await productDetailPage.getProductPrice()
    expect(price).toBeTruthy()
  })

  it('应该能够切换到商品描述', async () => {
    await productDetailPage.switchToDescription()
  })

  it('应该能够切换到商品规格', async () => {
    await productDetailPage.switchToSpecs()
  })

  it('应该能够切换到商品评价', async () => {
    await productDetailPage.switchToReviews()
  })

  it('应该能够收藏商品', async () => {
    await productDetailPage.toggleFavorite()
  })

  it('应该能够加入购物车', async () => {
    await productDetailPage.tapAddToCart()
  })

  it('应该能够立即购买', async () => {
    await productDetailPage.tapBuyNow()
    await checkoutPage.waitForLoad()
  })
})

describe('购物车功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
  })

  it('应该显示空购物车', async () => {
    await cartPage.waitForLoad()
    const isEmpty = await cartPage.isCartEmpty()
    expect(isEmpty).toBe(true)
  })

  it('应该能够选择购物车商品', async () => {
    await cartPage.selectItem(1)
  })

  it('应该能够全选商品', async () => {
    await cartPage.selectAll()
  })

  it('应该能够增加商品数量', async () => {
    await cartPage.increaseQuantity(1)
  })

  it('应该能够减少商品数量', async () => {
    await cartPage.decreaseQuantity(1)
  })

  it('应该能够删除购物车商品', async () => {
    await cartPage.deleteItem(1)
  })

  it('应该能够获取总价', async () => {
    const price = await cartPage.getTotalPrice()
    expect(price).toBeTruthy()
  })

  it('应该能够点击结算按钮', async () => {
    await cartPage.tapCheckout()
  })
})

describe('结算功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
  })

  it('应该能够加载结算页', async () => {
    await checkoutPage.waitForLoad()
  })

  it('应该能够选择收货地址', async () => {
    await checkoutPage.selectAddress(1)
  })

  it('应该能够选择支付方式', async () => {
    await checkoutPage.selectPayment('微信支付')
  })

  it('应该能够输入订单备注', async () => {
    await checkoutPage.enterRemark('请尽快发货')
  })

  it('应该能够获取总价', async () => {
    const price = await checkoutPage.getTotalPrice()
    expect(price).toBeTruthy()
  })

  it('应该能够提交订单', async () => {
    await checkoutPage.submitOrder()
  })
})

describe('订单功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
    await orderPage.waitForLoad()
  })

  it('应该能够切换到待付款', async () => {
    await orderPage.switchToPending()
  })

  it('应该能够切换到已付款', async () => {
    await orderPage.switchToPaid()
  })

  it('应该能够切换到已发货', async () => {
    await orderPage.switchToShipped()
  })

  it('应该能够切换到已完成', async () => {
    await orderPage.switchToCompleted()
  })

  it('应该能够查看订单详情', async () => {
    await orderPage.tapOrder(1)
  })

  it('应该能够获取订单数量', async () => {
    const count = await orderPage.getOrderCount()
    expect(count).toBeGreaterThanOrEqual(0)
  })
})

describe('用户中心功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
  })

  it('应该能够加载用户中心', async () => {
    await userPage.waitForLoad()
  })

  it('应该能够获取用户昵称', async () => {
    const nickname = await userPage.getNickname()
    expect(nickname).toBeTruthy()
  })

  it('应该能够进入设置页面', async () => {
    await userPage.tapSettings()
    await device.back()
  })

  it('应该能够进入订单页面', async () => {
    await userPage.tapOrders()
    await orderPage.waitForLoad()
    await device.back()
  })

  it('应该能够进入地址管理页面', async () => {
    await userPage.tapAddresses()
    await device.back()
  })

  it('应该能够进入收藏页面', async () => {
    await userPage.tapFavorites()
    await device.back()
  })

  it('应该能够进入优惠券页面', async () => {
    await userPage.tapCoupons()
    await device.back()
  })
})

describe('搜索功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
  })

  it('应该能够打开搜索页', async () => {
    await homePage.tapSearchBar()
    await searchPage.waitForLoad()
    const isVisible = await searchPage.visible
    expect(isVisible).toBe(true)
  })

  it('应该能够搜索商品', async () => {
    await searchPage.search('iPhone')
    const count = await searchPage.getResultCount()
    expect(count).toBeGreaterThanOrEqual(0)
  })

  it('应该能够点击搜索结果', async () => {
    await searchPage.tapResult(1)
    await productDetailPage.waitForLoad()
    await device.back()
  })

  it('应该能够清空搜索历史', async () => {
    await searchPage.clearHistory()
  })
})

describe('登录功能测试', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
  })

  it('应该能够打开登录页', async () => {
    await loginPage.waitForLoad()
  })

  it('应该验证登录按钮初始状态', async () => {
    const isEnabled = await loginPage.isLoginButtonEnabled()
    expect(isEnabled).toBe(false)
  })

  it('应该能够输入用户名和密码', async () => {
    await loginPage.login('testuser', 'password123')
  })

  it('应该能够跳转到注册页', async () => {
    await loginPage.tapRegister()
  })

  it('应该能够跳转到忘记密码页', async () => {
    await loginPage.tapForgotPassword()
  })
})

describe('应用状态测试', () => {
  it('应该能够在后台保持状态', async () => {
    await device.sendToHome()
    await device.launchApp({ fromIcon: true })
  })

  it('应该能够重新加载应用', async () => {
    await reload()
    await homePage.waitForLoad()
  })
})