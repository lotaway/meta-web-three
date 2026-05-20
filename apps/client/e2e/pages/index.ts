import { by, element, expect } from 'detox'

class Page {
  protected get visible(): Promise<boolean> {
    return this.root.isVisible()
  }

  async waitForVisible(timeout = 10000): Promise<void> {
    await this.root.waitForVisible(timeout)
  }

  async waitForHidden(timeout = 10000): Promise<void> {
    await this.root.waitForHidden(timeout)
  }

  async scrollTo(direction: 'up' | 'down' | 'left' | 'right'): Promise<void> {
    await this.root.scroll(direction)
  }
}

export class TabBar extends Page {
  private root = element(by.id('tab-bar'))

  async navigateToTab(tabName: string): Promise<void> {
    await element(by.text(tabName)).tap()
  }

  async isTabVisible(tabName: string): Promise<boolean> {
    return element(by.text(tabName)).isVisible()
  }
}

export class HomePage extends Page {
  private root = element(by.id('home-page'))
  private searchInput = element(by.id('search-input'))
  private categoryTab = element(by.id('category-tab'))
  private brandTab = element(by.id('brand-tab'))
  private productList = element(by.id('product-list'))
  private advertiseList = element(by.id('advertise-list'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async tapSearchBar(): Promise<void> {
    await this.searchInput.tap()
  }

  async scrollToBottom(): Promise<void> {
    await this.productList.scrollTo('bottom')
  }

  async tapProduct(productId: number): Promise<void> {
    await element(by.id(`product-${productId}`)).tap()
  }

  async tapCategoryTab(): Promise<void> {
    await this.categoryTab.tap()
  }

  async tapBrandTab(): Promise<void> {
    await this.brandTab.tap()
  }

  async getProductCount(): Promise<number> {
    const count = await this.productList.getAttributes()
    return count.children ? count.children.length : 0
  }
}

export class CategoryPage extends Page {
  private root = element(by.id('category-page'))
  private categoryList = element(by.id('category-list'))
  private subCategoryList = element(by.id('sub-category-list'))
  private productGrid = element(by.id('product-grid')

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async tapCategory(categoryId: number): Promise<void> {
    await element(by.id(`category-${categoryId}`)).tap()
  }

  async getCategoryNames(): Promise<string[]> {
    const elements = await this.categoryList.findAll(by.id('category-name'))
    return Promise.all(elements.map(el => el.getText()))
  }

  async scrollCategoryList(direction: 'up' | 'down'): Promise<void> {
    await this.categoryList.scroll(direction)
  }

  async tapProduct(productId: number): Promise<void> {
    await element(by.id(`product-${productId}`)).tap()
  }
}

export class ProductDetailPage extends Page {
  private root = element(by.id('product-detail-page'))
  private imageSwiper = element(by.id('product-images'))
  private nameText = element(by.id('product-name'))
  private priceText = element(by.id('product-price'))
  private addToCartButton = element(by.id('add-to-cart-btn'))
  private buyNowButton = element(by.id('buy-now-btn'))
  private favoriteButton = element(by.id('favorite-btn'))
  private shareButton = element(by.id('share-btn'))
  private descriptionTab = element(by.id('description-tab'))
  private specsTab = element(by.id('specs-tab'))
  private reviewsTab = element(by.id('reviews-tab'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async getProductName(): Promise<string> {
    return this.nameText.getText()
  }

  async getProductPrice(): Promise<string> {
    return this.priceText.getText()
  }

  async tapAddToCart(): Promise<void> {
    await this.addToCartButton.tap()
  }

  async tapBuyNow(): Promise<void> {
    await this.buyNowButton.tap()
  }

  async toggleFavorite(): Promise<void> {
    await this.favoriteButton.tap()
  }

  async tapShare(): Promise<void> {
    await this.shareButton.tap()
  }

  async switchToDescription(): Promise<void> {
    await this.descriptionTab.tap()
  }

  async switchToSpecs(): Promise<void> {
    await this.specsTab.tap()
  }

  async switchToReviews(): Promise<void> {
    await this.reviewsTab.tap()
  }

  async swipeImage(direction: 'left' | 'right'): Promise<void> {
    await this.imageSwiper.swipe(direction)
  }
}

export class CartPage extends Page {
  private root = element(by.id('cart-page'))
  private cartItemList = element(by.id('cart-item-list'))
  private checkoutButton = element(by.id('checkout-btn'))
  private selectAllButton = element(by.id('select-all-btn'))
  private totalPriceText = element(by.id('total-price'))
  private emptyCartView = element(by.id('empty-cart-view'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async isCartEmpty(): Promise<boolean> {
    return this.emptyCartView.isVisible()
  }

  async selectItem(itemId: number): Promise<void> {
    await element(by.id(`cart-item-${itemId}`)).tap()
  }

  async selectAll(): Promise<void> {
    await this.selectAllButton.tap()
  }

  async tapCheckout(): Promise<void> {
    await this.checkoutButton.tap()
  }

  async getTotalPrice(): Promise<string> {
    return this.totalPriceText.getText()
  }

  async increaseQuantity(itemId: number): Promise<void> {
    await element(by.id(`increase-${itemId}`)).tap()
  }

  async decreaseQuantity(itemId: number): Promise<void> {
    await element(by.id(`decrease-${itemId}`)).tap()
  }

  async deleteItem(itemId: number): Promise<void> {
    await element(by.id(`delete-${itemId}`)).swipe('left')
    await element(by.text('删除')).tap()
  }
}

export class CheckoutPage extends Page {
  private root = element(by.id('checkout-page'))
  private addressSection = element(by.id('address-section'))
  private paymentSection = element(by.id('payment-section'))
  private orderItemList = element(by.id('order-item-list'))
  private submitButton = element(by.id('submit-order-btn'))
  private totalPriceText = element(by.id('checkout-total-price'))
  private remarkInput = element(by.id('remark-input'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async selectAddress(addressId: number): Promise<void> {
    await element(by.id(`address-${addressId}`)).tap()
  }

  async selectPayment(method: string): Promise<void> {
    await element(by.text(method)).tap()
  }

  async enterRemark(text: string): Promise<void> {
    await this.remarkInput.replaceText(text)
  }

  async submitOrder(): Promise<void> {
    await this.submitButton.tap()
  }

  async getTotalPrice(): Promise<string> {
    return this.totalPriceText.getText()
  }
}

export class OrderPage extends Page {
  private root = element(by.id('order-page'))
  private tabBar = element(by.id('order-tab-bar'))
  private orderList = element(by.id('order-list'))
  private pendingTab = element(by.id('pending-tab'))
  private paidTab = element(by.id('paid-tab'))
  private shippedTab = element(by.id('shipped-tab'))
  private completedTab = element(by.id('completed-tab'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async switchToPending(): Promise<void> {
    await this.pendingTab.tap()
  }

  async switchToPaid(): Promise<void> {
    await this.paidTab.tap()
  }

  async switchToShipped(): Promise<void> {
    await this.shippedTab.tap()
  }

  async switchToCompleted(): Promise<void> {
    await this.completedTab.tap()
  }

  async tapOrder(orderId: number): Promise<void> {
    await element(by.id(`order-${orderId}`)).tap()
  }

  async getOrderCount(): Promise<number> {
    const list = await this.orderList.findAll(by.id('order-item'))
    return list.length
  }
}

export class OrderDetailPage extends Page {
  private root = element(by.id('order-detail-page'))
  private orderInfo = element(by.id('order-info'))
  private orderStatusText = element(by.id('order-status'))
  private payButton = element(by.id('pay-btn'))
  private cancelButton = element(by.id('cancel-btn'))
  private confirmReceiveButton = element(by.id('confirm-receive-btn'))
  private logisticsButton = element(by.id('logistics-btn'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async getOrderStatus(): Promise<string> {
    return this.orderStatusText.getText()
  }

  async tapPay(): Promise<void> {
    await this.payButton.tap()
  }

  async tapCancel(): Promise<void> {
    await this.cancelButton.tap()
  }

  async tapConfirmReceive(): Promise<void> {
    await this.confirmReceiveButton.tap()
  }

  async tapLogistics(): Promise<void> {
    await this.logisticsButton.tap()
  }
}

export class UserPage extends Page {
  private root = element(by.id('user-page'))
  private avatar = element(by.id('user-avatar'))
  private nicknameText = element(by.id('nickname'))
  private settingsButton = element(by.id('settings-btn'))
  private orderButton = element(by.id('orders-btn'))
  private addressButton = element(by.id('addresses-btn'))
  private favoriteButton = element(by.id('favorites-btn'))
  private couponButton = element(by.id('coupons-btn'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async getNickname(): Promise<string> {
    return this.nicknameText.getText()
  }

  async tapSettings(): Promise<void> {
    await this.settingsButton.tap()
  }

  async tapOrders(): Promise<void> {
    await this.orderButton.tap()
  }

  async tapAddresses(): Promise<void> {
    await this.addressButton.tap()
  }

  async tapFavorites(): Promise<void> {
    await this.favoriteButton.tap()
  }

  async tapCoupons(): Promise<void> {
    await this.couponButton.tap()
  }
}

export class AddressPage extends Page {
  private root = element(by.id('address-page'))
  private addressList = element(by.id('address-list'))
  private addButton = element(by.id('add-address-btn'))
  private editButton = element(by.id('edit-address-btn'))
  private deleteButton = element(by.id('delete-address-btn'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async addNewAddress(): Promise<void> {
    await this.addButton.tap()
  }

  async editAddress(addressId: number): Promise<void> {
    await element(by.id(`edit-${addressId}`)).tap()
  }

  async deleteAddress(addressId: number): Promise<void> {
    await element(by.id(`delete-${addressId}`)).swipe('left')
  }

  async setDefaultAddress(addressId: number): Promise<void> {
    await element(by.id(`set-default-${addressId}`)).tap()
  }

  async getAddressCount(): Promise<number> {
    const list = await this.addressList.findAll(by.id('address-item'))
    return list.length
  }
}

export class SearchPage extends Page {
  private root = element(by.id('search-page'))
  private searchInput = element(by.id('search-input'))
  private resultList = element(by.id('search-result-list'))
  private historyList = element(by.id('search-history'))
  private clearHistoryButton = element(by.id('clear-history-btn'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async search(keyword: string): Promise<void> {
    await this.searchInput.replaceText(keyword)
    await element(by.text('搜索')).tap()
  }

  async clearHistory(): Promise<void> {
    await this.clearHistoryButton.tap()
  }

  async tapResult(productId: number): Promise<void> {
    await element(by.id(`result-${productId}`)).tap()
  }

  async getResultCount(): Promise<number> {
    const list = await this.resultList.findAll(by.id('result-item'))
    return list.length
  }
}

export class LoginPage extends Page {
  private root = element(by.id('login-page'))
  private usernameInput = element(by.id('username-input'))
  private passwordInput = element(by.id('password-input'))
  private loginButton = element(by.id('login-btn'))
  private registerButton = element(by.id('register-btn'))
  private forgotPasswordButton = element(by.id('forgot-password-btn'))

  async waitForLoad(): Promise<void> {
    await this.root.waitForVisible()
  }

  async login(username: string, password: string): Promise<void> {
    await this.usernameInput.replaceText(username)
    await this.passwordInput.replaceText(password)
    await this.loginButton.tap()
  }

  async tapRegister(): Promise<void> {
    await this.registerButton.tap()
  }

  async tapForgotPassword(): Promise<void> {
    await this.forgotPasswordButton.tap()
  }

  async isLoginButtonEnabled(): Promise<boolean> {
    return this.loginButton.isEnabled()
  }
}