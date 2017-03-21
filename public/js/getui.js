/**
 * 异步请求
 * @param {Object} params 异步参数对象
 */
function request(params = {}) {
    if (!params.url) return null;
    var xhr = new XMLHttpRequest(), //	异步对象
        responseText = {}; //	响应内容
    params.type = params.type || 'POST';
    params.data = params.data || '';
    params.dataType = params.dataType || 'JSON';
    if (params.timeout) xhr.timeout = params.timeout;
    xhr.onreadystatechange = function () {
        if (typeof params.change == 'function') params.change(xhr.readyState);
        if (xhr.readyState == 4) {
            // console.info(params.url + "  " + xhr.readyState + "  " + xhr.status);
            if (xhr.status == 200) {
                switch (params.dataType) {
                    case 'JSON':
                        try {
                            responseText = JSON.parse(xhr.responseText);
                        }
                        catch (e) {
                            responseText = xhr.responseText;
                        }
                        break;
                    default:
                        responseText = xhr.responseText;
                        break;
                }
                if (typeof params.success === 'function') {
                    params.success(responseText);
                }
            } else {
                if (typeof params.error === 'function') params.error(xhr);
            }
            if (typeof params.complete === 'function') {
                params.complete(responseText, xhr);
            }
        }
    };
    xhr.open(params.type, params.url);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.send(params.data);
    return xhr;
}

window.addEventListener("load", function () {

    document.getElementById("btn_submit").addEventListener("click", function (e) {

        var queryObj = {
                title: "标题"
                , content: "内容"
            }
            , queryData = ""
            ;

        for (var name in queryObj) {
            queryData += "&" + name + "=" + (document.getElementById(name).value || queryObj[name]);
        }

        request({
            url: "/api/getui",
            data: queryData.replace(/^&/, ''),
            success: function (data) {
                console.log(data);
            },
            error: function (error) {
                alert(error);
            }
        });

    });

});