/**
 * 🧾 DOMAIN EVENT BASE CLASS
 *
 * Represents a single immutable domain event.
 */
class DomainEvent {
  constructor(eventName, payload = {}) {
    if (!eventName) {
      throw new Error('DomainEvent requires an event name');
    }

    this.eventName = eventName;
    this.payload = payload;
    this.occurredAt = new Date();
  }
}

module.exports = { DomainEvent };
