/**
 * 🛰️ In-memory Event Bus
 *
 * Lightweight event bus for domain events.
 * Swap with Kafka/Rabbit for production.
 */
class InMemoryEventBus {
  constructor() {
    this.handlers = new Map();
  }

  subscribe(eventName, handler) {
    if (!this.handlers.has(eventName)) {
      this.handlers.set(eventName, []);
    }

    this.handlers.get(eventName).push(handler);
  }

  publish(event) {
    const handlers = this.handlers.get(event.eventName) || [];
    for (const handler of handlers) {
      handler(event);
    }
  }
}

module.exports = { InMemoryEventBus };
