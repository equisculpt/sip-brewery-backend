const { InMemoryEventBus } = require('./InMemoryEventBus');

const eventBus = new InMemoryEventBus();

module.exports = { eventBus, InMemoryEventBus };
