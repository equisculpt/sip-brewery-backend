/**
 * 💰 MONEY VALUE OBJECT
 */
class Money {
  constructor(amount, currency = 'INR') {
    if (amount === null || amount === undefined || Number.isNaN(amount)) {
      throw new Error('Money requires a valid amount');
    }

    this.amount = Number(amount);
    this.currency = currency;
  }

  add(other) {
    this.assertSameCurrency(other);
    return new Money(this.amount + other.amount, this.currency);
  }

  subtract(other) {
    this.assertSameCurrency(other);
    return new Money(this.amount - other.amount, this.currency);
  }

  isLessThan(other) {
    this.assertSameCurrency(other);
    return this.amount < other.amount;
  }

  toString() {
    return `${this.currency} ${this.amount.toFixed(2)}`;
  }

  assertSameCurrency(other) {
    if (this.currency !== other.currency) {
      throw new Error('Currency mismatch');
    }
  }
}

module.exports = { Money };
