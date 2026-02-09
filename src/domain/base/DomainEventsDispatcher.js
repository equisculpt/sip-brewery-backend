/**
 * 📡 Domain Events Dispatcher
 */
const { eventBus } = require('../../infrastructure/eventBus');

class DomainEventsDispatcher {
  static dispatchEvents(aggregate) {
    const events = aggregate.pullDomainEvents();
    events.forEach((event) => eventBus.publish(event));
  }
}

module.exports = { DomainEventsDispatcher };
