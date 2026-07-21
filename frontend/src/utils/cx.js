/**
 * Joins class name segments into a single space-separated string, filtering out falsy values.
 * Useful for conditional class application in React elements.
 * 
 * @param {...(string|boolean|null|undefined)} classes - Class name strings or conditional expressions.
 * @returns {string} The joined class list.
 */
export function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}
export default cx;
