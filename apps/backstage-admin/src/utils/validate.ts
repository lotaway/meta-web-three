export function isvalidUsername(str: string) {
  return str.trim().length >= 3
}

/* Valid URI */
export function validateURL(textval: string) {
  const urlregex =
    /^(https?|ftp):\/\/([a-zA-Z0-9.-]+(:[a-zA-Z0-9.&%$-]+)*@)*((25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}|([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.(com|edu|gov|int|mil|net|org|biz|arpa|info|name|pro|aero|coop|museum|[a-zA-Z]{2}))(:[0-9]+)*(\/($|[a-zA-Z0-9.,?'\\+&%$#=~_-]+))*$/
  return urlregex.test(textval)
}

/* Lowercase letters */
export function validateLowerCase(str: string) {
  const reg = /^[a-z]+$/
  return reg.test(str)
}

/* Uppercase letters */
export function validateUpperCase(str: string) {
  const reg = /^[A-Z]+$/
  return reg.test(str)
}

/* Mixed-case letters */
export function validatAlphabets(str: string) {
  const reg = /^[A-Za-z]+$/
  return reg.test(str)
}
