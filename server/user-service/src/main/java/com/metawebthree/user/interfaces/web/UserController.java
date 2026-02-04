package com.metawebthree.user.interfaces.web;

import com.metawebthree.author.AuthorDO;
import com.metawebthree.common.constants.RequestHeaderKeys;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.generated.rpc.GetOrderByUserIdRequest;
import com.metawebthree.common.generated.rpc.OrderDTO;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.utils.UserRole;
import com.metawebthree.user.application.dto.LoginResponseDTO;
import com.metawebthree.user.application.dto.SubTokenDTO;
import com.metawebthree.user.application.dto.UserDTO;
import com.metawebthree.user.application.UserService;
import com.metawebthree.common.utils.DateEnum;
import com.metawebthree.common.utils.OAuth1Utils;
import com.metawebthree.common.utils.UserJwtUtil;

import lombok.extern.slf4j.Slf4j;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.security.GeneralSecurityException;
import java.security.NoSuchAlgorithmException;
import java.security.SignatureException;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.web3j.crypto.Credentials;
import org.web3j.crypto.Sign;
import org.web3j.utils.Numeric;

@Slf4j
@RestController
@RequestMapping("/user")
public class UserController {

    private final UserService userService;

    @Value("${x.apikey:未配置}")
    private String twitterApiKey;

    @Value("${x.secretkey:未配置}")
    private String twitterSecretKey;

    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    private final UserJwtUtil jwtUtil;

    public UserController(UserService userService, UserJwtUtil jwtUtil) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
    }

    @GetMapping("/list")
    public ApiResponse<List<UserDTO>> userList(@RequestParam(defaultValue = "1", required = false) Integer pageNum,
            @RequestParam(required = false) String email, @RequestParam(required = false) Long userRoleId,
            @RequestParam(required = false) String realName) {
        UserDTO userDTO = new UserDTO();
        userDTO.setUserRoleId(UserRole.valueOf(userRoleId));
        userDTO.setEmail(email);
        AuthorDO authorDO = new AuthorDO();
        authorDO.setRealName(realName);
        // return ApiResponse.success(userService.getUserList(pageNum,
        // wrapper).getRecords());
        return ApiResponse.success(userService.getUserList(pageNum, userDTO, authorDO).getRecords());
    }

    @PostMapping("/create")
    public ApiResponse<?> create(@RequestBody Map<String, Object> params) throws Exception {
        UserRole userRoleId = UserRole.tryValueOf((long) (params.get("typeId"))).orElse(UserRole.USER);
        Long referrerId = null;
        if (params.get("referrerId") != null) {
            try {
                referrerId = Long.parseLong(String.valueOf(params.get("referrerId")));
            } catch (NumberFormatException ex) {
                return ApiResponse.error(ResponseStatus.PARAM_TYPE_ERROR, "Invalid referrerId");
            }
            if (referrerId > 0 && userService.getById(referrerId) == null) {
                return ApiResponse.error(ResponseStatus.USER_NOT_FOUND, "Referrer not found");
            }
        }
        Long userId = userService.createUserWithReferrer(
                String.valueOf(params.get("email")),
                String.valueOf(params.get("password")), userRoleId, referrerId);
        log.info("New user created - userId: {}", userId);
        return ApiResponse.success();
    }

    @RequestMapping("/signIn")
    public ApiResponse<LoginResponseDTO> signIn(@RequestParam(defaultValue = "0", required = false) Long userRoleId,
            @RequestParam String email, @RequestParam String password,
            @RequestParam(defaultValue = "-1", required = false) Integer expiresInHours)
            throws IOException, NoSuchAlgorithmException {
        UserDTO user = userService.validateUser(email, password, userRoleId);
        if (user == null) {
            return ApiResponse.error(ResponseStatus.USER_PASSWORD_ERROR, "Invalid credentials");
        }

        Map<String, Object> claims = buildClaims(user);
        String token = generateToken(user.getId().toString(), claims, expiresInHours);

        return ApiResponse.success(new LoginResponseDTO(token, user, null, "email"));
    }

    private Map<String, Object> buildClaims(UserDTO user) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        claims.put("name", user.getNickname());
        claims.put("role", UserRole.USER.name());
        return claims;
    }

    private String generateToken(String userId, Map<String, Object> claims, Integer expiresInHours) {
        if (expiresInHours == -1) {
            return jwtUtil.generate(userId, claims);
        }
        if (expiresInHours == 0) {
            Date expiration = DateEnum.ONE_HUNDRED_YEAR.toAfterThisAsDate();
            return jwtUtil.generate(userId, claims, expiration);
        }
        Date expiration = new Date(System.currentTimeMillis() + expiresInHours * DateEnum.ONE_HOUR.getValue());
        return jwtUtil.generate(userId, claims, expiration);
    }

    @PostMapping("/signInWithWallet")
    public ApiResponse<LoginResponseDTO> signInWithWallet(@RequestParam String walletAddress,
            @RequestParam String timestamp, @RequestParam String signature)
            throws SignatureException, NoSuchAlgorithmException {
        String message = String.format("%s%s%s", walletAddress, "|login by wallet|", timestamp);

        byte[] signatureBytes = Numeric.hexStringToByteArray(signature);
        byte v = signatureBytes[64];
        if (v < 27) {
            v += 27;
        }
        Sign.SignatureData signatureData = new Sign.SignatureData(v,
                Numeric.toBytesPadded(Numeric.toBigInt(signature.substring(2, 66)), 32),
                Numeric.toBytesPadded(Numeric.toBigInt(signature.substring(66, 130)), 32));
        byte[] publicKey = Sign.signedMessageToKey(message.getBytes(), signatureData).toByteArray();
        Credentials credentials = Credentials.create(Numeric.toHexStringNoPrefix(publicKey));

        if (!credentials.getAddress().equalsIgnoreCase(walletAddress)) {
            return ApiResponse.error(ResponseStatus.USER_WALLET_MISMATCH, "Invalid signature");
        }

        UserDTO user = userService.findOrCreateUserByWallet(walletAddress);

        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        claims.put("walletAddress", walletAddress);
        claims.put("userRole", user.getUserRoleId());
        claims.put("role", "USER");

        String token = jwtUtil.generate(user.getId().toString(), claims);

        return ApiResponse.success(new LoginResponseDTO(token, user, walletAddress, "wallet"));
    }

    @GetMapping("/checkWeb3SignerMessage")
    public ApiResponse<String> checkSignerMessage(@RequestParam String walletAddress, @RequestParam String timestamp,
            @RequestParam String signature) throws SignatureException {
        String message = String.format("%s%s%s", walletAddress, "|login by wallet|", timestamp);

        byte[] signatureBytes = Numeric.hexStringToByteArray(signature);
        byte v = signatureBytes[64];
        if (v < 27) {
            v += 27;
        }
        Sign.SignatureData signatureData = new Sign.SignatureData(v,
                Numeric.toBytesPadded(Numeric.toBigInt(signature.substring(2, 66)), 32),
                Numeric.toBytesPadded(Numeric.toBigInt(signature.substring(66, 130)), 32));
        byte[] publicKey = Sign.signedMessageToKey(message.getBytes(), signatureData).toByteArray();
        Credentials credentials = Credentials.create(Numeric.toHexStringNoPrefix(publicKey));

        return ApiResponse.success(credentials.getAddress());
    }

    @GetMapping("/checkBitcoinSignature")
    public ApiResponse<?> checkSignerMessage2() {
        // PublicKey publicKey = new PublicKey("na");
        return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.SYSTEM_ERROR, "todo");
    }

    @GetMapping("/checkOAuth1")
    public String checkOAuth1() throws GeneralSecurityException, UnsupportedEncodingException {
        String method = "POST";
        String url = "https://api.twitter.com/oauth/request_token";
        String callback = "http://tsp.nat300.top/airdrop/bindAccount";
        Map<String, String> params = OAuth1Utils.getBaseOauth1Map(twitterApiKey, twitterSecretKey);
        params.put("oauth_callback", callback);
        String signature = OAuth1Utils.generate(method, url, params, twitterApiKey, null);
        params.put("oauth_signature", signature);

        StringBuilder queryStringBuilder = new StringBuilder();
        for (Map.Entry<String, String> param : params.entrySet()) {
            if (!queryStringBuilder.isEmpty()) {
                queryStringBuilder.append("&");
            }
            queryStringBuilder.append(OAuth1Utils.encode(param.getKey())).append("=")
                    .append(OAuth1Utils.encode(param.getValue()));
        }

        return url + "?" + queryStringBuilder.toString();
    }

    @GetMapping("/order")
    public ApiResponse<List<OrderDTO>> getOrderByUser(@RequestHeader Map<String, String> header,
            @RequestHeader(value = "Authorization", required = false) String authorization) {
        Long userId = extractUserId(authorization, header);
        GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder().setId(userId).build();
        List<OrderDTO> result = orderService.getOrderByUserId(request).getOrdersList();
        return ApiResponse.success(result);
    }

    private Long extractUserId(String authorization, Map<String, String> header) {
        if (authorization != null && authorization.startsWith("Bearer ")) {
            String token = authorization.substring("Bearer ".length());
            Optional<io.jsonwebtoken.Claims> oClaims = jwtUtil.tryDecode(token);
            if (oClaims.isPresent()) {
                return jwtUtil.getUserId(oClaims.get());
            }
        }
        Optional<String> oUserId = Optional.ofNullable(header.get(RequestHeaderKeys.USER_ID.getValue()));
        return Long.parseLong(oUserId.orElse("0"));
    }

    @PostMapping("/createSubToken")
    public ApiResponse<SubTokenDTO> createSubToken(@RequestHeader("Authorization") String authorizationHeader,
            @RequestParam(required = false) List<String> permissions,
            @RequestParam(defaultValue = "24") Long expiresInHours) {
        try {
            String parentToken = authorizationHeader.replace("Bearer ", "");
            if (expiresInHours == 0) {
                expiresInHours = DateEnum.ONE_HUNDRED_YEAR.getValue();
            }
            SubTokenDTO subToken = userService.createSubToken(parentToken, permissions, expiresInHours);
            return ApiResponse.success(subToken);
        } catch (Exception e) {
            log.error("Failed to create sub-token", e);
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR);
        }
    }
}
