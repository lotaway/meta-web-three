package com.metawebthree.user;

import com.metawebthree.author.AuthorDO;
import com.metawebthree.common.OAuth1Utils;
import com.metawebthree.common.contants.RequestHeaderKeys;
import com.metawebthree.common.dto.OrderDTO;
import com.metawebthree.common.rpc.interfaces.OrderService;
import com.metawebthree.common.utils.UserRole;
import com.metawebthree.user.DO.UserDO;
import com.metawebthree.user.DTO.LoginResponseDTO;
import com.metawebthree.user.impl.UserServiceImpl;
import com.metawebthree.common.ApiResponse;
import com.metawebthree.common.utils.JwtUtil;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import lombok.extern.slf4j.Slf4j;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.security.GeneralSecurityException;
import java.security.Key;
import java.security.NoSuchAlgorithmException;
import java.security.SignatureException;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.web3j.crypto.Credentials;
import org.web3j.crypto.Sign;
import org.web3j.utils.Numeric;

@Slf4j
@RestController
@RequestMapping("/user")
public class UserController {

    private final String subject = "metawebthree";

    // @Autowired
    private final UserServiceImpl userService;

    @DubboReference
    private OrderService orderService;

    private final JwtUtil jwtUtil;

    public UserController(UserServiceImpl userService, JwtUtil jwtUtil) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
    }

    @GetMapping("/list")
    public ApiResponse<List<UserDO>> userList(@RequestParam(defaultValue = "1", required = false) Integer pageNum,
            @RequestParam(required = false) String email, @RequestParam(required = false) Short typeId,
            @RequestParam(required = false) String realName) {
        UserDO userPojo = new UserDO();
        userPojo.setTypeId(typeId);
        userPojo.setEmail(email);
        AuthorDO authorPojo = new AuthorDO();
        authorPojo.setRealName(realName);
        // return ApiResponse.success(userService.getUserList(pageNum,
        // wrapper).getRecords());
        return ApiResponse.success(userService.getUserList(pageNum, userPojo, authorPojo).getRecords());
    }

    @PostMapping("/create")
    public ApiResponse<?> create(@RequestBody Map<String, Object> params) throws NoSuchAlgorithmException {
        Short typeId = (Short) params.get("typeId");
        if (typeId == null)
            typeId = 0;
        Long userId = userService.createUser(String.valueOf(params.get("email")),
                String.valueOf(params.get("password")), typeId);
        LocalDateTime localDateTime = LocalDateTime.now();
        log.info("userId:" + userId + ", date_time:" + localDateTime);
        return ApiResponse.success();
    }

    @RequestMapping("/signIn")
    public ApiResponse<LoginResponseDTO> signIn(@RequestParam(defaultValue = "0", required = false) Short typeId,
            @RequestParam String email, @RequestParam String password) throws IOException, NoSuchAlgorithmException {
        UserDO user = userService.validateUser(email, password, typeId);
        if (user == null) {
            return ApiResponse.error("Invalid credentials", LoginResponseDTO.class);
        }
        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        claims.put("name", user.getNickname());
        claims.put("role", UserRole.USER.name());
        String token = jwtUtil.generate(user.getId().toString(), claims);
        LoginResponseDTO response = new LoginResponseDTO(token, user, null, "email");
        return ApiResponse.success(response);
    }

    /**
     * Web3钱包登录
     */
    @PostMapping("/signInWithWallet")
    public ApiResponse<LoginResponseDTO> signInWithWallet(@RequestParam String walletAddress,
            @RequestParam String timestamp, @RequestParam String signature)
            throws SignatureException, NoSuchAlgorithmException {
        // 验证签名
        String message = String.format("%s%s%s", walletAddress, "|login by wallet|", timestamp);

        // 以太坊签名信息解析
        byte[] signatureBytes = Numeric.hexStringToByteArray(signature);
        byte v = signatureBytes[64];
        if (v < 27) {
            v += 27;
        }
        Sign.SignatureData signatureData = new Sign.SignatureData(v,
                Numeric.toBytesPadded(Numeric.toBigInt(signature.substring(2, 66)), 32),
                Numeric.toBytesPadded(Numeric.toBigInt(signature.substring(66, 130)), 32));

        // 从签名信息中提取公钥
        byte[] publicKey = Sign.signedMessageToKey(message.getBytes(), signatureData).toByteArray();
        // 从公钥创建凭证
        Credentials credentials = Credentials.create(Numeric.toHexStringNoPrefix(publicKey));

        // 验证钱包地址是否匹配
        if (!credentials.getAddress().equalsIgnoreCase(walletAddress)) {
            return ApiResponse.error("Invalid signature", LoginResponseDTO.class);
        }

        // 查找或创建用户
        UserDO user = userService.findOrCreateUserByWallet(walletAddress);

        // 生成JWT token
        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        claims.put("walletAddress", walletAddress);
        claims.put("typeId", user.getTypeId());
        claims.put("role", "USER");

        String token = jwtUtil.generate(user.getId().toString(), claims);

        // 返回登录成功信息和token
        LoginResponseDTO response = new LoginResponseDTO(token, user, walletAddress, "wallet");

        return ApiResponse.success(response);
    }

    @RequestMapping("/checkAuth")
    public ApiResponse<?> checkAuth(@RequestParam String jwt) {
        Key key = Keys.secretKeyFor(SignatureAlgorithm.HS256);
        Claims claims = Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(jwt).getBody();
        assert claims.getSubject().equals(subject);
        return ApiResponse.success(claims);
    }

    @GetMapping("/checkWeb3SignerMessage")
    public ApiResponse<String> checkSignerMessage(@RequestParam String walletAddress, @RequestParam String timestamp,
            @RequestParam String signature) throws SignatureException {
        // 要验证的消息
        String message = String.format("%s%s%s", walletAddress, "|login by wallet|", timestamp); // 替换为您的消息

        // 以太坊签名信息解析
        byte[] signatureBytes = Numeric.hexStringToByteArray(signature);
        byte v = signatureBytes[64];
        if (v < 27) {
            v += 27;
        }
        Sign.SignatureData signatureData = new Sign.SignatureData(v,
                Numeric.toBytesPadded(Numeric.toBigInt(signature.substring(2, 66)), 32),
                Numeric.toBytesPadded(Numeric.toBigInt(signature.substring(66, 130)), 32));
        // 从签名信息中提取公钥
        byte[] publicKey = Sign.signedMessageToKey(message.getBytes(), signatureData).toByteArray();
        // 从公钥创建凭证
        Credentials credentials = Credentials.create(Numeric.toHexStringNoPrefix(publicKey));

        return ApiResponse.success(credentials.getAddress());
    }

    @GetMapping("/checkBitcoinSignature")
    public ApiResponse<?> checkSignerMessage2() {
        // PublicKey publicKey = new PublicKey("na");
        return ApiResponse.error("todo");
    }

    @GetMapping("/checkOAuth1")
    public String checkOAuth1() throws GeneralSecurityException, UnsupportedEncodingException {
        String twitterApiKey = "yDdyzOFkLHmmn5tJ6NeCqbSDy";
        String twitterSecretKey = "z3skm6f9Hb3RNx9ltfmMj6Vm9LTBXaqJhMYCTdmnNOxpOFHwyp";
        String method = "POST";
        String url = "https://api.twitter.com/oauth/request_token";
        String callback = "http://tsp.nat300.top/airdrop/bindAccount";
        Map<String, String> params = OAuth1Utils.getBaseOauth1Map(twitterApiKey, twitterSecretKey);
        params.put("oauth_callback", callback);
        String signature = OAuth1Utils.generate(method, url, params, twitterApiKey, null);
        params.put("oauth_signature", signature);

        StringBuilder queryStringBuilder = new StringBuilder();
        for (Map.Entry<String, String> param : params.entrySet()) {
            if (queryStringBuilder.length() > 0) {
                queryStringBuilder.append("&");
            }
            queryStringBuilder.append(OAuth1Utils.encode(param.getKey())).append("=")
                    .append(OAuth1Utils.encode(param.getValue()));
        }

        return url + "?" + queryStringBuilder.toString();
    }

    @GetMapping("/order")
    public ApiResponse<List<OrderDTO>> getOrderByUser(@RequestHeader Map<String, String> header) {
        Long userId = Long.parseLong(header.get(RequestHeaderKeys.USER_ID.getValue()));
        return ApiResponse.success(orderService.getOrderByUserId(userId));
    }

}
