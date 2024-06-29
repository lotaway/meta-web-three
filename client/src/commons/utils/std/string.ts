import { Tail } from "./types";

type ToShortParams = Tail<Parameters<typeof to_short>>

export class String {
    constructor(private str: string) {
    }

    to_short(...args: ToShortParams): string {
        return to_short(this.str, ...args)
    }
}

export function to_short(str: string, begin: number = 0, end: number = 10): string {
    return str.slice(begin, end) + '...';
}