package com.metawebthree.cart.interfaces.web;

import com.metawebthree.cart.application.CartService;
import com.metawebthree.common.constants.RequestHeaderKeys;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.dto.CartItemDTO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Validated
@RestController
@RequestMapping("/v1/cart")
@RequiredArgsConstructor
@Tag(name = "Cart Management", description = "Endpoints for shopping cart lifecycle and management")
public class CartController {

    private final CartService cartService;

    @Operation(summary = "Add item to cart")
    @PostMapping("/add")
    public ApiResponse<Integer> add(@RequestHeader(RequestHeaderKeys.USER_ID_VALUE) Long userId,
                                @RequestBody CartItemDTO cartItem) {
        cartItem.setMemberId(userId);
        return ApiResponse.success(cartService.add(cartItem));
    }

    @Operation(summary = "List current user's cart items")
    @GetMapping("/list")
    public ApiResponse<List<CartItemDTO>> list(@RequestHeader(RequestHeaderKeys.USER_ID_VALUE) Long userId) {
        return ApiResponse.success(cartService.list(userId));
    }

    @Operation(summary = "Update cart item quantity")
    @PutMapping("/update/quantity")
    public ApiResponse<Integer> updateQuantity(@RequestHeader(RequestHeaderKeys.USER_ID_VALUE) Long userId,
                                             @RequestParam Long id,
                                             @RequestParam Integer quantity) {
        return ApiResponse.success(cartService.updateQuantity(userId, id, quantity));
    }

    @Operation(summary = "Remove items from cart")
    @DeleteMapping("/delete")
    public ApiResponse<Integer> delete(@RequestHeader(RequestHeaderKeys.USER_ID_VALUE) Long userId,
                                     @RequestParam("ids") List<Long> ids) {
        return ApiResponse.success(cartService.delete(userId, ids));
    }

    @Operation(summary = "Clear current user's cart")
    @PostMapping("/clear")
    public ApiResponse<Integer> clear(@RequestHeader(RequestHeaderKeys.USER_ID_VALUE) Long userId) {
        return ApiResponse.success(cartService.clear(userId));
    }
}
